#include <iostream>

#include "tga/tga.hpp"
#include "tga/tga_createInfo_structs.hpp"
#include "tga/tga_utils.hpp"

struct Light {
    alignas(16) glm::vec3 lightPos = glm::vec3(0);
    alignas(16) glm::vec4 lightColor = glm::vec4(0);
};

struct Camera {
    alignas(16) glm::mat4 toWorld = glm::mat4(1); // camera world pos (inverse of view matrix) -> needed for phong specular component
    alignas(16) glm::mat4 view = glm::mat4(1);
    alignas(16) glm::mat4 projection = glm::mat4(1);

    Camera(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up, float fov, float aspectRatio, float near,
        float far)
    {
        projection = glm::perspective_vk(fov, aspectRatio, near, far);
        view = glm::lookAt(position, target, up);
        toWorld = glm::inverse(view);
    }
};

struct Transform {
    alignas(16) glm::mat4 transform = glm::mat4(1); //model world pos (model matrix)
};

template <typename T>
tga::Buffer makeBufferFromStruct(tga::Interface& tgai, tga::BufferUsage usage, T& data) {
    auto size = sizeof(T);
    auto staging = tgai.createStagingBuffer({size, tga::memoryAccess(data)});
    auto buffer = tgai.createBuffer({usage, size, staging});
    tgai.free(staging);
    return buffer;
};

auto makeBufferFromVector = [&](tga::Interface& tgai, tga::BufferUsage usage, auto& vec) {
    auto size = vec.size() * sizeof(vec[0]);
    auto staging = tgai.createStagingBuffer({size, tga::memoryAccess(vec)});
    auto buffer = tgai.createBuffer({usage, size, staging});
    tgai.free(staging);
    return buffer;
};


int main()
{
    std::cout << "GPU Pro\n";
    tga::Interface tgai;

    // create window
    auto windowWidth = tgai.screenResolution().first / 6 * 5;
    auto windowHeight = tgai.screenResolution().second / 6 * 5;
    tga::Window window = tgai.createWindow({windowWidth, windowHeight});

    //load the shaders
    const std::string vertexShaderRelPath = "../shaders/deferredRendering_vert.spv";
    const std::string fragShaderRelPath = "../shaders/deferredRendering_frag.spv";
    tga::Shader vertexShader = tga::loadShader(vertexShaderRelPath, tga::ShaderType::vertex, tgai);
    tga::Shader fragShader = tga::loadShader(fragShaderRelPath, tga::ShaderType::fragment, tgai);

    //load OBJ "man" that has inside vertex buffer and index buffer 
    tga::Obj obj_man = tga::loadObj("../../../Data/man/man.obj");
    std::vector<tga::Vertex> vBuffer_man = obj_man.vertexBuffer;
    std::vector<uint32_t> iBuffer_man = obj_man.indexBuffer;

    // load OBJ "transporter" that has inside vertex buffer and index buffer
    tga::Obj obj_transporter = tga::loadObj("../../../Data/transporter/transporter.obj");
    std::vector<tga::Vertex> vBuffer_transporter = obj_transporter.vertexBuffer;
    std::vector<uint32_t> iBuffer_transporter = obj_transporter.indexBuffer;

    //load "man" diffuse texture data 
    const std::string diffuseTexRelPath_man = "../../../Data/man/man_diffuse.png";
    tga::TextureBundle diffuseTex_man = tga::loadTexture(diffuseTexRelPath_man, 
                                        tga::Format::r8g8b8a8_srgb, tga::SamplerMode::nearest, tgai);

    // load "transporter" diffuse texture data
    const std::string diffuseTexRelPath_transporter = "../../../Data/transporter/transporter_diffuse.png";
    tga::TextureBundle diffuseTex_transporter = tga::loadTexture(diffuseTexRelPath_transporter, 
                                                tga::Format::r8g8b8a8_srgb, tga::SamplerMode::nearest, tgai);    

    //Define camera position and orientation such that meshes are in view, i.e. define view, proj matrices and inv of view to get camera pos
    const glm::vec3 position = glm::vec3(0.f, 7.f, 11.f);
    const glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);
    const glm::vec3 cameraTarget(0);
    float aspectRatio = windowWidth / static_cast<float>(windowHeight);
    auto fov = glm::radians(60.f);
    Camera cam(position, cameraTarget, up, fov, aspectRatio, 0.1f, 1000.f);

    //"man" model 
    tga::Buffer vertexBuffer_man = makeBufferFromVector(tgai, tga::BufferUsage::vertex, vBuffer_man);
    tga::Buffer indexBuffer_man = makeBufferFromVector(tgai, tga::BufferUsage::index, iBuffer_man);
  
    //create buffer which stores array of transforms for "man" instances
    std::vector<Transform> transformsBuffer_man;
    int nInstances_man = 40;
    int rowSize = 10;
    int nRows = nInstances_man / rowSize;
    for (int j = 0; j < nRows; ++j) {
        for (int i = 0; i < rowSize; ++i) {
            glm::vec3 worldPos = glm::vec3(-10.0f + 2 * i, 0.0f, -4.0f - 2 * j);
            Transform objTransform;
            objTransform.transform = glm::translate(glm::mat4(1.0f), worldPos) * glm::scale(glm::mat4(1), glm::vec3(0.01));
            transformsBuffer_man.push_back({objTransform});
        }
    }
    tga::Buffer transformData_man = makeBufferFromVector(tgai, tga::BufferUsage::uniform, transformsBuffer_man);
    
    //"transporter" model
    tga::Buffer vertexBuffer_transporter = makeBufferFromVector(tgai, tga::BufferUsage::vertex, vBuffer_transporter);
    tga::Buffer indexBuffer_transporter = makeBufferFromVector(tgai, tga::BufferUsage::index, iBuffer_transporter);

    //create buffer which stores array of transforms for "transporter" instances
    std::vector<Transform> transformsBuffer_transporter;
    int nInstances_transporter = 10;
    for (int i = 0; i < nInstances_man; ++i) {
        glm::vec3 worldPos = glm::vec3(-10.0f + 2*i, 0.0f, 2.0f);
        Transform objTransform;
        float angle = glm::radians(270.0f); //90° rotation...
        glm::vec3 axis = glm::vec3(0.0f, 1.0f, 0.0f); //...around Y axis
        objTransform.transform = glm::translate(glm::mat4(1), worldPos) *
                                 glm::scale(glm::mat4(1), glm::vec3(0.005)) * glm::rotate(glm::mat4(1), angle, axis);
        transformsBuffer_transporter.push_back({objTransform});
    }
    tga::Buffer transformData_transporter = makeBufferFromVector(tgai, tga::BufferUsage::uniform, transformsBuffer_transporter);

    //create inputLayout for the renderPass
    tga::InputLayout inputLayout({
        // Set = 0: Camera data, Light data
        {tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer}, 
        // Set = 1: Diffuse tex, Transform data
        {tga::BindingType::uniformBuffer, tga::BindingType::sampler}
        });


    // create renderPass
    tga::RenderPassInfo renderPassInfo(vertexShader, fragShader, window);
    renderPassInfo.setClearOperations(tga::ClearOperation::all)
        .setPerPixelOperations(tga::PerPixelOperations{}.setDepthCompareOp(tga::CompareOperation::lessEqual))
        .setVertexLayout(tga::Vertex::layout())
        .setInputLayout(inputLayout);
    tga::RenderPass renderPass = tgai.createRenderPass(renderPassInfo);


    //create buffer to send camera data(view + proj) to GPU
    tga::Buffer cameraData = makeBufferFromStruct(tgai, tga::BufferUsage::uniform, cam);

    //initialize many lights, assigning different positions and eventually colors
    std::vector<Light> lights;
    int nLigths = 2;
    lights.push_back({glm::vec3(2., 3., -1.), glm::vec4(glm::vec3(0, 0, 1), 1)});
    lights.push_back({glm::vec3(-2., 3., -1.), glm::vec4(glm::vec3(1, 0, 0), 1)});
    tga::Buffer lightsBuffer = makeBufferFromVector(tgai, tga::BufferUsage::uniform, lights);
    //tga::InputSet inputSetLight = tgai.createInputSet({renderPass, {tga::Binding{lightsBuffer, 0}}, 1});

    tga::InputSet inputSetCameraLight =
        tgai.createInputSet({renderPass, {tga::Binding{cameraData, 0}, tga::Binding{lightsBuffer, 1}}, 0});

    //create input set for specific mesh data (position and textures)
    //"man"
    tga::InputSet inputSet_man =
        tgai.createInputSet({renderPass, {tga::Binding{transformData_man, 0}, tga::Binding{diffuseTex_man, 1}}, 1});

    //"transporter"
    tga::InputSet inputSet_transporter = 
        tgai.createInputSet({renderPass, {tga::Binding{transformData_transporter, 0}, tga::Binding{diffuseTex_transporter, 1}}, 1});

    //instantiate a commandBuffer
    std::vector<tga::CommandBuffer> cmdBuffers(tgai.backbufferCount(window));
    tga::CommandBuffer cmdBuffer{};
    
    //rendering loop
    while (!tgai.windowShouldClose(window)) {
        //handle to the frameBuffer of the window
        uint32_t nextFrame = tgai.nextFrame(window);
        tga::CommandBuffer& cmdBuffer = cmdBuffers[nextFrame];
        if (!cmdBuffer) {
            //initialize a commandRecorder to start recording commands
            tga::CommandRecorder cmdRecorder = tga::CommandRecorder{tgai, cmdBuffer};
            //setup the cmd recorder by passing it a render pass, the bindings and the draw calls
            cmdRecorder.setRenderPass(renderPass, nextFrame, {0., 0., 0., 1}).bindInputSet(inputSetCameraLight);

            cmdRecorder.bindVertexBuffer(vertexBuffer_man)
                .bindIndexBuffer(indexBuffer_man)
                .bindInputSet(inputSet_man)
                .drawIndexed(iBuffer_man.size(), 0, 0, nInstances_man, 0)
                .bindVertexBuffer(vertexBuffer_transporter)
                .bindIndexBuffer(indexBuffer_transporter)
                .bindInputSet(inputSet_transporter)
                .drawIndexed(iBuffer_transporter.size(), 0, 0, nInstances_transporter, 0);
                
            // the command recorder has done recording and can initialize a commandBuffer
            cmdBuffer = cmdRecorder.endRecording();

        } 
        else {
            // Need to reset the command buffer before re-using it
            tgai.waitForCompletion(cmdBuffer);
        }
        
        //execute the commands recorded in the commandBuffer
        tgai.execute(cmdBuffer);
        //present the current data in the frameBuffer "nextFrame" to the window
        tgai.present(window, nextFrame);
    }

    return 0;
}
