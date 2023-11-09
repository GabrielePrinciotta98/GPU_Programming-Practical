#include <iostream>

#include "tga/tga.hpp"
#include "tga/tga_createInfo_structs.hpp"
#include "tga/tga_utils.hpp"

struct Light {
    alignas(16) glm::vec3 lightPos = glm::vec3(0, 2.0, 0);
    alignas(16) glm::vec4 lightColor = glm::vec4(0.9, 0.9, 0.9, 1);
};

struct Camera {
    alignas(16) glm::mat4 toWorld = glm::mat4(1); // camera world pos (inverse of view matrix) -> needed for phong specular component
    alignas(16) glm::mat4 view = glm::mat4(1);
    alignas(16) glm::mat4 projection = glm::mat4(1);
};

struct Transform {
    alignas(16) glm::mat4 transform = glm::mat4(1); //model world pos (model matrix)
};


int main()
{
    std::cout << "GPU Pro\n";
    tga::Interface tgai;

    // create window
    auto windowWidth = tgai.screenResolution().first / 2;
    auto windowHeight = tgai.screenResolution().second / 2;
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
    tga::TextureBundle diffuseTex_transporter =
        tga::loadTexture(diffuseTexRelPath_transporter, tga::Format::r8g8b8a8_srgb, tga::SamplerMode::nearest, tgai);

    //Assign a unique position to each mesh, defining a model matrix
    glm::vec3 worldPos_man = glm::vec3(-5.0f, 1.0f, 1.0f);
    Transform objTransform_man;
    objTransform_man.transform = glm::translate(glm::mat4(1.0f), worldPos_man) * glm::scale(glm::mat4(1), glm::vec3(0.02)); 

    glm::vec3 worldPos_transporter = glm::vec3(1.0f, -1.0f, -5.0f);
    Transform objTransform_transporter;
    objTransform_transporter.transform =
        glm::translate(glm::mat4(1.0f), worldPos_transporter) * glm::scale(glm::mat4(1), glm::vec3(0.01));

    //Define camera position and orientation such that meshes are in view, i.e. define view, proj matrices and inv of view to get camera pos
    Camera cam;
    const glm::vec3 position = glm::vec3(5.f, 5.f, 5.f);
    const glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);
    float aspectRatio = windowWidth / static_cast<float>(windowHeight);
    auto fov = glm::radians(60.f);
    cam.projection = glm::perspective_vk(fov, aspectRatio, 0.1f, 1000.f);
    cam.view = glm::lookAt(position, glm::vec3(0, 0, 0), up);
    cam.toWorld = glm::inverse(cam.view);

    //create buffer for mesh specific data: vertex buffer, index buffer and transform
    //"man" model 
    size_t vertexBufferSize_man = vBuffer_man.size() * sizeof(tga::Vertex);
    size_t indexBufferSize_man = iBuffer_man.size() * sizeof(uint32_t);

    tga::StagingBuffer stagingBufferVertex_man = tgai.createStagingBuffer({vertexBufferSize_man, tga::memoryAccess(vBuffer_man)});
    tga::Buffer vertexBuffer_man = tgai.createBuffer({tga::BufferUsage::vertex, vertexBufferSize_man, stagingBufferVertex_man});
    tgai.free(stagingBufferVertex_man);

    tga::StagingBuffer stagingBufferIndex_man = tgai.createStagingBuffer({indexBufferSize_man, tga::memoryAccess(iBuffer_man)});
    tga::Buffer indexBuffer_man = tgai.createBuffer({tga::BufferUsage::index, indexBufferSize_man, stagingBufferIndex_man});
    tgai.free(stagingBufferIndex_man);

    tga::StagingBuffer stagingBufferTransform_man = tgai.createStagingBuffer({sizeof(objTransform_man), tga::memoryAccess(objTransform_man)});
    tga::Buffer transformData_man = tgai.createBuffer({tga::BufferUsage::uniform, sizeof(objTransform_man), stagingBufferTransform_man});
    tgai.free(stagingBufferTransform_man);

    //"transporter" model
    size_t vertexBufferSize_transporter = vBuffer_transporter.size() * sizeof(tga::Vertex);
    size_t indexBufferSize_transporter = iBuffer_transporter.size() * sizeof(uint32_t);

    tga::StagingBuffer stagingBufferVertex_transporter = tgai.createStagingBuffer({vertexBufferSize_transporter, tga::memoryAccess(vBuffer_transporter)});
    tga::Buffer vertexBuffer_transporter = tgai.createBuffer({tga::BufferUsage::vertex, vertexBufferSize_transporter, stagingBufferVertex_transporter});
    tgai.free(stagingBufferVertex_transporter);

    tga::StagingBuffer stagingBufferIndex_transporter = tgai.createStagingBuffer({indexBufferSize_transporter, tga::memoryAccess(iBuffer_transporter)});
    tga::Buffer indexBuffer_transporter = tgai.createBuffer({tga::BufferUsage::index, indexBufferSize_transporter, stagingBufferIndex_transporter});
    tgai.free(stagingBufferIndex_transporter);

    tga::StagingBuffer stagingBufferTransform_transporter = tgai.createStagingBuffer({sizeof(objTransform_transporter), tga::memoryAccess(objTransform_transporter)});
    tga::Buffer transformData_transporter =
        tgai.createBuffer({tga::BufferUsage::uniform, sizeof(objTransform_transporter), stagingBufferTransform_transporter});
    tgai.free(stagingBufferTransform_transporter);


    //create inputLayout for the renderPass
    tga::InputLayout inputLayout({
        // Set = 0: Camera data, Light data
        {tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer},
        // Set = 1: Diffuse tex, Transform data
        {tga::BindingType::sampler, tga::BindingType::uniformBuffer}
        });


    // create renderPass
    tga::RenderPassInfo renderPassInfo(vertexShader, fragShader, window);
    renderPassInfo.setClearOperations(tga::ClearOperation::all)
        .setPerPixelOperations(tga::PerPixelOperations{}.setDepthCompareOp(tga::CompareOperation::lessEqual))
        .setVertexLayout(tga::Vertex::layout())
        .setInputLayout(inputLayout);
    tga::RenderPass renderPass = tgai.createRenderPass(renderPassInfo);


    //create buffers to send camera data(view + proj) and light data to the GPU
    tga::StagingBuffer stagingBufferCamera = tgai.createStagingBuffer({sizeof(cam), tga::memoryAccess(cam)});
    tga::Buffer cameraData = tgai.createBuffer({tga::BufferUsage::uniform, sizeof(cam), stagingBufferCamera});
    tgai.free(stagingBufferCamera);


    Light light;
    tga::StagingBuffer stagingBufferLight = tgai.createStagingBuffer({sizeof(light), tga::memoryAccess(light)});
    tga::Buffer lightData = tgai.createBuffer({tga::BufferUsage::uniform, sizeof(light), stagingBufferLight});
    tgai.free(stagingBufferLight);

    //create input set for uniform data to all meshes (camera and light)
    tga::InputSet inputSetUniform =
        tgai.createInputSet({renderPass, {tga::Binding{cameraData, 0}, tga::Binding{lightData, 1}}, 0});

    //create input set for specific mesh data (position and textures)
    //"man"
    tga::InputSet inputSet_man =
        tgai.createInputSet({renderPass, {tga::Binding{diffuseTex_man, 0}, tga::Binding{transformData_man, 1}}, 1});

    //"transporter"
    tga::InputSet inputSet_transporter = 
        tgai.createInputSet({renderPass, {tga::Binding{diffuseTex_transporter, 0}, tga::Binding{transformData_transporter, 1}}, 1});

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
            cmdRecorder.setRenderPass(renderPass, nextFrame)
                .bindInputSet(inputSetUniform)
                .bindVertexBuffer(vertexBuffer_man)
                .bindIndexBuffer(indexBuffer_man)
                .bindInputSet(inputSet_man)
                .drawIndexed(iBuffer_man.size(), 0, 0)
                .bindVertexBuffer(vertexBuffer_transporter)
                .bindIndexBuffer(indexBuffer_transporter)
                .bindInputSet(inputSet_transporter)
                .drawIndexed(iBuffer_transporter.size(), 0, 0);
                
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
