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
    const std::string vertexShaderRelPath = "../shaders/simpleForwardRendering_vert.spv";
    const std::string fragShaderRelPath = "../shaders/simpleForwardRendering_frag.spv";
    
    tga::Shader vertexShader = tga::loadShader(vertexShaderRelPath, tga::ShaderType::vertex, tgai);
    tga::Shader fragShader = tga::loadShader(fragShaderRelPath, tga::ShaderType::fragment, tgai);

    //load OBJ that has inside vertex buffer and index buffer 
    tga::Obj obj_man = tga::loadObj("../../Data/man/man.obj");
    std::vector<tga::Vertex> vBuffer_man = obj_man.vertexBuffer;
    std::vector<uint32_t> iBuffer_man = obj_man.indexBuffer;

    //load texture data
    const std::string diffuseTexRelPath_man = "../../Data/man/man_diffuse.png";
    tga::TextureBundle diffuseTex_man = tga::loadTexture(diffuseTexRelPath_man, 
                                        tga::Format::r8g8b8a8_srgb, tga::SamplerMode::nearest, tgai);

    //Assign a unique position to each mesh, defining a model matrix
    glm::vec3 objWorldPos = glm::vec3(1.0f, 1.0f, 1.0f);
    Transform objTransform_man;
    objTransform_man.transform = glm::translate(glm::mat4(1.0f), objWorldPos) * glm::scale(glm::mat4(1), glm::vec3(0.01));  

    //Define camera position and orientation such that meshes are in view, i.e. define view, proj matrices and inv of view to get camera pos
    Camera cam;
    const glm::vec3 position = glm::vec3(5.f, 5.f, 5.f);
    const glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);
    float aspectRatio = windowWidth / static_cast<float>(windowHeight);
    cam.projection = glm::perspective_vk(glm::radians(90.f), aspectRatio, 0.1f, 1000.f);
    cam.view = glm::lookAt(position, glm::vec3(0, 0, 0), up);
    cam.toWorld = glm::inverse(cam.view);

    //create a vertex buffer and an index buffer to send meshes data to GPU
    size_t vertexBufferSize = vBuffer_man.size() * sizeof(tga::Vertex);
    size_t indexBufferSize = iBuffer_man.size() * sizeof(uint32_t);

    tga::StagingBuffer stagingBufferVertex_man = tgai.createStagingBuffer(
        {
            vertexBufferSize, 
            tga::memoryAccess(vBuffer_man)
        }
    );
    
    tga::Buffer vertexBuffer_man = tgai.createBuffer(
        {
            tga::BufferUsage::vertex, 
            vertexBufferSize, 
            stagingBufferVertex_man
        }
    );
    tgai.free(stagingBufferVertex_man);


    tga::StagingBuffer stagingBufferIndex_man = tgai.createStagingBuffer(
        {
            indexBufferSize, 
            tga::memoryAccess(iBuffer_man)
        }
    );
    
    tga::Buffer indexBuffer_man = tgai.createBuffer(
        {
            tga::BufferUsage::index, 
            indexBufferSize, 
            stagingBufferIndex_man
        }
    );
    tgai.free(stagingBufferIndex_man);

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


    //create uniform buffers to send transform data (model matrix), camera data(view + proj) and light data to the GPU
    tga::StagingBuffer stagingBufferTransform_man = tgai.createStagingBuffer(
        {
            sizeof(objTransform_man), 
            tga::memoryAccess(objTransform_man)
        }
    );
    
    tga::Buffer transformData_man = tgai.createBuffer(
        {
            tga::BufferUsage::uniform, 
            sizeof(objTransform_man), 
            stagingBufferTransform_man
        }
    );
    tgai.free(stagingBufferTransform_man);

    tga::StagingBuffer stagingBufferCamera = tgai.createStagingBuffer(
        {
            sizeof(cam), 
            tga::memoryAccess(cam)
        }
    );
    
    tga::Buffer cameraData = tgai.createBuffer(
        {
            tga::BufferUsage::uniform, 
            sizeof(cam), 
            stagingBufferCamera
        }
    );
    tgai.free(stagingBufferCamera);


    Light light;
    tga::StagingBuffer stagingBufferLight = tgai.createStagingBuffer(
        {
            sizeof(light), 
            tga::memoryAccess(light)
            //reinterpret_cast<uint8_t const*>(&light)
        }
    );
    
    tga::Buffer lightData = tgai.createBuffer(
        {
            tga::BufferUsage::uniform, 
            sizeof(light), 
            stagingBufferLight
        }
    );
    tgai.free(stagingBufferLight);

    //create input set for uniform data to all meshes (camera and light)
    tga::InputSet inputSetUniform = tgai.createInputSet(
        {
            renderPass, 
            {tga::Binding{cameraData, 0}, tga::Binding{lightData, 1}},
            0
        }
    );

    //create input set for specific mesh data (position and textures)
    tga::InputSet inputSet_man = tgai.createInputSet(
        {
            renderPass, 
            {tga::Binding{diffuseTex_man, 0}, tga::Binding{transformData_man, 1}},
            1
        }
    );

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
                .bindVertexBuffer(vertexBuffer_man)
                .bindIndexBuffer(indexBuffer_man)
                .bindInputSet(inputSet_man)
                .bindInputSet(inputSetUniform)
                .drawIndexed(iBuffer_man.size(), 0, 0);
                
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
