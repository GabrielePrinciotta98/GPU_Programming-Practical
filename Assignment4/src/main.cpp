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

    const std::string vertexShader_defRendering1_Path = "../shaders/deferredRenderingFirstPass_vert.spv";
    const std::string fragShader_defRendering1_Path = "../shaders/deferredRenderingFirstPass_frag.spv";
    const std::string vertexShader_defRendering2_Path = "../shaders/deferredRenderingSecondPass_vert.spv";
    const std::string fragShader_defRendering2_Path = "../shaders/deferredRenderingSecondPass_frag.spv";
    tga::Shader vertexShaderFirstPass = tga::loadShader(vertexShader_defRendering1_Path, tga::ShaderType::vertex, tgai);
    tga::Shader fragShaderFirstPass = tga::loadShader(fragShader_defRendering1_Path, tga::ShaderType::fragment, tgai);   
    tga::Shader vertexShaderSecondPass = tga::loadShader(vertexShader_defRendering2_Path, tga::ShaderType::vertex, tgai);
    tga::Shader fragShaderSecondPass = tga::loadShader(fragShader_defRendering2_Path, tga::ShaderType::fragment, tgai);

    const std::string vertexShader_TexToScreen_Path = "../shaders/renderTextureToScreen_vert.spv";
    const std::string fragShader_PostProcessing_Path = "../shaders/postProcessing_frag.spv";
    tga::Shader vertexShaderTexToScreen = tga::loadShader(vertexShader_TexToScreen_Path, tga::ShaderType::vertex, tgai);
    tga::Shader fragShaderPostProc = tga::loadShader(fragShader_PostProcessing_Path, tga::ShaderType::fragment, tgai);
    

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
    //const glm::vec3 position = glm::vec3(0.f, 5.f, 11.f);
    const glm::vec3 position = glm::vec3(10.f, 2.f, 5.f);
    
    const glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);
    const glm::vec3 cameraTarget = glm::vec3(0, 0, -7);
    float aspectRatio = windowWidth / static_cast<float>(windowHeight);
    auto fov = glm::radians(60.f);
    Camera cam(position, cameraTarget, up, fov, aspectRatio, 0.1f, 1000.f);
    // create buffer to send camera data(view + proj) to GPU
    tga::Buffer cameraData = makeBufferFromStruct(tgai, tga::BufferUsage::uniform, cam);


    //"man" model 
    tga::Buffer vertexBuffer_man = makeBufferFromVector(tgai, tga::BufferUsage::vertex, vBuffer_man);
    tga::Buffer indexBuffer_man = makeBufferFromVector(tgai, tga::BufferUsage::index, iBuffer_man);
  
    //create buffer which stores array of transforms for "man" instances
    std::vector<Transform> transformsBuffer_man;
    int nInstances_man = 44;
    int rowSize = 11;
    int nRows = nInstances_man / rowSize;
    for (int j = 0; j < nRows; ++j) {
        for (int i = 0; i < rowSize; ++i) {
            glm::vec3 worldPos = glm::vec3(-10.0f + 2 * i, 0.0f, -4.0f - 3 * j);
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
    int nInstances_transporter = 11;
    for (int i = 0; i < nInstances_man; ++i) {
        glm::vec3 worldPos = glm::vec3(-10.0f + 2*i, 0.0f, 2.0f);
        Transform objTransform;
        float angle = glm::radians(270.0f); //270° rotation...
        glm::vec3 axis = glm::vec3(0.0f, 1.0f, 0.0f); //...around Y axis
        objTransform.transform = glm::translate(glm::mat4(1), worldPos) *
                                 glm::scale(glm::mat4(1), glm::vec3(0.005)) * glm::rotate(glm::mat4(1), angle, axis);
        transformsBuffer_transporter.push_back({objTransform});
    }
    tga::Buffer transformData_transporter = makeBufferFromVector(tgai, tga::BufferUsage::uniform, transformsBuffer_transporter);

    
    // initialize many lights, assigning different positions and eventually colors
    std::vector<Light> lights;
    int nLigths = 144;
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            lights.push_back(
                {glm::vec3(30, 30 - 2 * i, 5 - 2 * j), glm::vec4(glm::vec3(0.01 + 0.002 * i, 0.01 + 0.002 * j, 0.005 + i * 0.001), 1)});
        }
    }

    tga::Buffer lightsBuffer = makeBufferFromVector(tgai, tga::BufferUsage::uniform, lights);


    // Data for a quad (quad vertex buffer)
    std::vector<glm::vec2> quadVertices = {glm::vec2(-1, -1), glm::vec2(1, 1), glm::vec2(-1, 1),
                                           glm::vec2(1, -1)};
    tga::Buffer vertexBuffer_quad = makeBufferFromVector(tgai, tga::BufferUsage::vertex, quadVertices);

    // Indices for a quad, aka 2 triangles (quad index buffer)
    std::vector<uint32_t> quadIndices = {0, 1, 2, 0, 3, 1};
    tga::Buffer indexBuffer_quad = makeBufferFromVector(tgai, tga::BufferUsage::index, quadIndices);

   
    // create inputLayout for the first renderPass
    tga::InputLayout inputLayoutFirstPass({// Set = 0: Camera data
                                           {tga::BindingType::uniformBuffer},
                                           // Set = 1: Transform data, Diffuse Tex
                                           {tga::BindingType::uniformBuffer, tga::BindingType::sampler}
                                         });

    // create first renderPass : input loaded data -> output g-buffer (tex list)
    std::vector<tga::Texture> gBufferData;
    tga::Texture fragWorldPositions = tgai.createTexture({windowWidth, windowHeight, tga::Format::r32g32b32a32_sfloat});
    tga::Texture normals = tgai.createTexture({windowWidth, windowHeight, tga::Format::r32g32b32a32_sfloat});
    tga::Texture albedo = tgai.createTexture({windowWidth, windowHeight, tga::Format::r32g32b32a32_sfloat});
    gBufferData.push_back(fragWorldPositions);
    gBufferData.push_back(normals);
    gBufferData.push_back(albedo);
    tga::Buffer gBuffer = makeBufferFromVector(tgai, tga::BufferUsage::uniform, gBufferData);
        
    tga::RenderPassInfo renderPassInfoFirstPass(vertexShaderFirstPass, fragShaderFirstPass, gBufferData);
    renderPassInfoFirstPass.setClearOperations(tga::ClearOperation::all)
        .setPerPixelOperations(tga::PerPixelOperations{}.setDepthCompareOp(tga::CompareOperation::lessEqual))
        .setVertexLayout(tga::Vertex::layout())
        .setInputLayout(inputLayoutFirstPass);
    tga::RenderPass renderPassFirst = tgai.createRenderPass(renderPassInfoFirstPass);


 
    
    tga::InputSet inputSetCameraFirst =
        tgai.createInputSet({renderPassFirst, {tga::Binding{cameraData, 0}}, 0});

    // create input set for specific mesh data (position and textures)
    //"man"
    tga::InputSet inputSet_man = tgai.createInputSet(
        {renderPassFirst, {tga::Binding{transformData_man, 0}, tga::Binding{diffuseTex_man, 1}}, 1});

    //"transporter"
    tga::InputSet inputSet_transporter = tgai.createInputSet(
        {renderPassFirst, {tga::Binding{transformData_transporter, 0}, tga::Binding{diffuseTex_transporter, 1}}, 1});


    // create inputLayout for the second renderPass
    tga::InputLayout inputLayoutSecondPass({
            // Set = 0: Camera data, Light data
            {tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer},
            // Set = 1 : fragWorldpos, normals, albedo
            {tga::BindingType::sampler, tga::BindingType::sampler, tga::BindingType::sampler}
        });

     // create second renderPass : input g-buffer (tex list) -> output intermediateResult (tex)
    tga::Texture intermediateResult = tgai.createTexture({windowWidth, windowHeight, tga::Format::r32g32b32a32_sfloat});
    tga::RenderPassInfo renderPassInfoSecond(vertexShaderSecondPass, fragShaderSecondPass, intermediateResult);
    renderPassInfoSecond.setClearOperations(tga::ClearOperation::all)
        .setInputLayout(inputLayoutSecondPass)
        .setVertexLayout(tga::VertexLayout{sizeof(glm::vec2), {{0, tga::Format::r32g32_sfloat}}});
    tga::RenderPass renderPassSecond = tgai.createRenderPass(renderPassInfoSecond);
    
    // create inputSets for the second renderPass
    tga::InputSet inputSetCameraLightSecond =
        tgai.createInputSet({renderPassSecond, {tga::Binding{cameraData, 0}, tga::Binding{lightsBuffer, 1}}, 0});
    tga::InputSet inputSetGBuffer = 
        tgai.createInputSet({renderPassSecond,
                             {tga::Binding{fragWorldPositions, 0}, tga::Binding{normals, 1}, tga::Binding{albedo, 2}},
                             1});

    // create inputLayout for the third renderPass (post-proc)
    tga::InputLayout inputLayoutPost({{tga::BindingType::sampler}});  // Set = 0 : RenderedTex
    
    // create third renderPass : input intermediateResult (tex) -> output window
    tga::RenderPassInfo renderPassInfoPost(vertexShaderTexToScreen, fragShaderPostProc, window);
    renderPassInfoPost.setInputLayout(inputLayoutPost)
        .setVertexLayout(
            tga::VertexLayout{sizeof(glm::vec2), {{0, tga::Format::r32g32_sfloat}}});
    tga::RenderPass renderPassPost = tgai.createRenderPass(renderPassInfoPost);
    
     // create inputSets for the third renderPass
    tga::InputSet inputSetIntermediateRes = tgai.createInputSet({renderPassPost, {tga::Binding{intermediateResult, 0}}, 0});



    //instantiate a commandBuffer
    std::vector<tga::CommandBuffer> cmdBuffers(tgai.backbufferCount(window));
    tga::CommandBuffer cmdBuffer{};
    
    // rendering loop
    while (!tgai.windowShouldClose(window)) {
        // handle to the frameBuffer of the window
        uint32_t nextFrame = tgai.nextFrame(window);
        tga::CommandBuffer& cmdBuffer = cmdBuffers[nextFrame];
        if (!cmdBuffer) {
            // initialize a commandRecorder to start recording commands
            tga::CommandRecorder cmdRecorder = tga::CommandRecorder{tgai, cmdBuffer};
            // setup the cmd recorder by passing it a render pass, the bindings and the draw calls
            cmdRecorder.setRenderPass(renderPassFirst, 0).bindInputSet(inputSetCameraFirst);

            cmdRecorder.bindVertexBuffer(vertexBuffer_man)
                .bindIndexBuffer(indexBuffer_man)
                .bindInputSet(inputSet_man)
                .drawIndexed(iBuffer_man.size(), 0, 0, nInstances_man, 0)
                .bindVertexBuffer(vertexBuffer_transporter)
                .bindIndexBuffer(indexBuffer_transporter)
                .bindInputSet(inputSet_transporter)
                .drawIndexed(iBuffer_transporter.size(), 0, 0, nInstances_transporter, 0);

            cmdRecorder.setRenderPass(renderPassSecond, 0)
                .bindInputSet(inputSetCameraLightSecond)
                .bindInputSet(inputSetGBuffer)
                .bindVertexBuffer(vertexBuffer_quad)
                .bindIndexBuffer(indexBuffer_quad)
                .drawIndexed(6, 0, 0);

            cmdRecorder.setRenderPass(renderPassPost, nextFrame)
                .bindInputSet(inputSetIntermediateRes)
                .bindVertexBuffer(vertexBuffer_quad)
                .bindIndexBuffer(indexBuffer_quad)
                .drawIndexed(6, 0, 0);

            // the command recorder has done recording and can initialize a commandBuffer
            cmdBuffer = cmdRecorder.endRecording();

        } else {
            // Need to reset the command buffer before re-using it
            tgai.waitForCompletion(cmdBuffer);
        }

        // execute the commands recorded in the commandBuffer
        tgai.execute(cmdBuffer);
        // present the current data in the frameBuffer "nextFrame" to the window
        tgai.present(window, nextFrame);
    }

    return 0;
}
