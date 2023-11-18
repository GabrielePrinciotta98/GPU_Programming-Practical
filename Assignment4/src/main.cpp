#include <iostream>
#include <filesystem>
#include <span>
#include <chrono>
#include <fstream>

#include "tga/tga.hpp"
#include "tga/tga_createInfo_structs.hpp"
#include "tga/tga_utils.hpp"

#include "CameraController.hpp"
namespace fs = std::filesystem;


struct Light {
    alignas(16) glm::vec3 lightPos = glm::vec3(0);
    alignas(16) glm::vec4 lightColor = glm::vec4(0);
};


struct Transform {
    alignas(16) glm::mat4 transform = glm::mat4(1); //model world pos (model matrix)
};


struct ConfigData {
    glm::vec3 pos{0, 0, 0}, offsets{0, 0, 0};
    float scale{0};
    uint32_t amount{0};
};

//data for each model
struct ModelData {
    ConfigData cfg;
    tga::Buffer modelMatrices;
    tga::Buffer vertexBuffer, indexBuffer;
    uint32_t indexCount{0};
    tga::Texture colorTex;
};


// A little helper function to create a staging buffer that acts like a specific type
template <typename T>
std::tuple<T&, tga::StagingBuffer, size_t> stagingBufferOfType(tga::Interface& tgai)
{
    auto stagingBuff = tgai.createStagingBuffer({sizeof(T)});
    return {*static_cast<T *>(tgai.getMapping(stagingBuff)), stagingBuff, sizeof(T)};
}

template <typename T>
tga::Buffer makeBufferFromStruct(tga::Interface& tgai, tga::BufferUsage usage, T& data) {
    auto size = sizeof(T);
    auto staging = tgai.createStagingBuffer({size, tga::memoryAccess(data)});
    auto buffer = tgai.createBuffer({usage, size, staging});
    tgai.free(staging);
    return buffer;
};

auto makeBufferFromVector = [](tga::Interface& tgai, tga::BufferUsage usage, auto& vec) {
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


    #pragma region create window
    auto windowWidth = tgai.screenResolution().first / 6 * 5;
    auto windowHeight = tgai.screenResolution().second / 6 * 5;
    tga::Window window = tgai.createWindow({windowWidth, windowHeight});
    #pragma endregion


    #pragma region load shaders
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
    #pragma endregion


    #pragma region load model data
    std::unordered_map<std::filesystem::path, ModelData> modelData;
    //std::cout << std::filesystem::current_path() << std::endl;
    for (auto entry : std::filesystem::directory_iterator{std::filesystem::current_path() / "input"}) {
        if (!entry.is_regular_file()) continue;
        std::ifstream config(entry.path());
        if (!config.is_open()) continue;

        auto extension = entry.path().extension();
        if (extension == ".config") {
            /*
             * Config file is in this format:
             *
            posX posY posZ
            offX offY offZ
            scale
            instanceAmount
             *
             Delimited by white space
             */
            auto& cfg = modelData[entry.path().stem()].cfg;
            config >> cfg.pos.x >> cfg.pos.y >> cfg.pos.z;
            config >> cfg.offsets.x >> cfg.offsets.y >> cfg.offsets.z;
            config >> cfg.scale;
            config >> cfg.amount;
        } else if (extension == ".obj") {
            auto obj = tga::loadObj(entry.path().string());
            auto makeBuffer = [&](tga::BufferUsage usage, auto& vec) {
                auto size = vec.size() * sizeof(vec[0]);
                auto staging = tgai.createStagingBuffer({size, tga::memoryAccess(vec)});
                auto buffer = tgai.createBuffer({usage, size, staging});
                tgai.free(staging);
                return buffer;
            };

            auto& data = modelData[entry.path().stem()];

            data.vertexBuffer = makeBuffer(tga::BufferUsage::vertex, obj.vertexBuffer);
            data.indexBuffer = makeBuffer(tga::BufferUsage::index, obj.indexBuffer);
            data.indexCount = obj.indexBuffer.size();
        } else {
            auto stem = entry.path().stem().string();
            auto pos = stem.find('_');
            pos = pos != std::string::npos ? pos : stem.size() - 1;  // Make sure to not crash on out of bounds access
            auto modelName = std::string_view{stem}.substr(0, pos);
            auto fileType = std::string_view{stem}.substr(pos + 1);
            if (fileType == "diffuse")
                modelData[modelName].colorTex =
                    tga::loadTexture(entry.path().string(), tga::Format::r8g8b8a8_srgb, tga::SamplerMode::linear, tgai);
            // TODO: Support different types of texturess
        }
    }
    #pragma endregion


    #pragma region create storage buffers for the model matrices of the instances for each model
    for (auto& [modelName, data] : modelData) {
        auto matrixStaging = tgai.createStagingBuffer({sizeof(glm::mat4) * data.cfg.amount});
        std::span<glm::mat4> matrixData{static_cast<glm::mat4 *>(tgai.getMapping(matrixStaging)), data.cfg.amount};
        for (size_t i = 0; i < matrixData.size(); ++i) {
            auto& matrix = matrixData[i];
            glm::vec3 position = data.cfg.pos;
            position += float(i) * data.cfg.offsets;
            matrix = glm::translate(glm::mat4(1), position) * glm::scale(glm::mat4(1), glm::vec3(data.cfg.scale));
        }
        data.modelMatrices = tgai.createBuffer({tga::BufferUsage::storage, matrixData.size_bytes(), matrixStaging});
        tgai.free(matrixStaging);
    }
    #pragma endregion


    #pragma region initialize camera controller and create camera buffer
    const glm::vec3 startPosition = glm::vec3(0.f, 2.f, 0.f);
    float aspectRatio = windowWidth / static_cast<float>(windowHeight);
    std::unique_ptr<CameraController> camController = std::make_unique<CameraController>(tgai, window, 60, aspectRatio, 0.1f, 30000.f, startPosition,
                                                       glm::vec3{0, 0, 1}, glm::vec3{0, 1, 0});
    tga::Buffer cameraData =
        tgai.createBuffer(tga::BufferInfo{tga::BufferUsage::uniform, sizeof(Camera), camController->Data()});
    #pragma endregion


    #pragma region initialize lights, assigning different positions and colors
    std::vector<Light> lights;
    int nLigths = 144;
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            lights.push_back(
                {glm::vec3(30, 30 - 2 * i, 5 - 2 * j), glm::vec4(glm::vec3(0.01 + 0.002 * i, 0.01 + 0.002 * j, 0.005 + i * 0.001), 1)});
        }
    }

    tga::Buffer lightsBuffer = makeBufferFromVector(tgai, tga::BufferUsage::uniform, lights);
    #pragma endregion


    #pragma region create vBuffer and iBuffer for quad needed for render to tex
    // Data for a quad (quad vertex buffer)
    std::vector<glm::vec2> quadVertices = {glm::vec2(-1, -1), glm::vec2(1, 1), glm::vec2(-1, 1),
                                           glm::vec2(1, -1)};
    tga::Buffer vertexBuffer_quad = makeBufferFromVector(tgai, tga::BufferUsage::vertex, quadVertices);

    // Indices for a quad, aka 2 triangles (quad index buffer)
    std::vector<uint32_t> quadIndices = {0, 1, 2, 0, 3, 1};
    tga::Buffer indexBuffer_quad = makeBufferFromVector(tgai, tga::BufferUsage::index, quadIndices);
    #pragma endregion


    #pragma region 1st renderPass initialization 
    // create inputLayout for the first renderPass
    tga::InputLayout inputLayoutGeometryPass({// Set = 0: Camera data
                                           {tga::BindingType::uniformBuffer},
                                           // Set = 1: Transform data, Diffuse Tex
                                           {tga::BindingType::storageBuffer, tga::BindingType::sampler}
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
        
    tga::RenderPassInfo geometryPassInfo(vertexShaderFirstPass, fragShaderFirstPass, gBufferData);
    geometryPassInfo.setClearOperations(tga::ClearOperation::all)
        .setPerPixelOperations(tga::PerPixelOperations{}.setDepthCompareOp(tga::CompareOperation::lessEqual))
        .setVertexLayout(tga::Vertex::layout())
        .setInputLayout(inputLayoutGeometryPass);
    tga::RenderPass geometryPass = tgai.createRenderPass(geometryPassInfo);
    #pragma endregion
   

    #pragma region 2nd renderPass initialization
    // create inputLayout for the second renderPass
    tga::InputLayout inputLayoutLightingPass({
            // Set = 0: Camera data, Light data
            {tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer},
            // Set = 1 : fragWorldpos, normals, albedo
            {tga::BindingType::sampler, tga::BindingType::sampler, tga::BindingType::sampler}
        });

     // create second renderPass : input g-buffer (tex list) -> output intermediateResult (tex)
    tga::Texture intermediateResult = tgai.createTexture({windowWidth, windowHeight, tga::Format::r32g32b32a32_sfloat});
    tga::RenderPassInfo lightingPassInfo(vertexShaderSecondPass, fragShaderSecondPass, intermediateResult);
    lightingPassInfo.setClearOperations(tga::ClearOperation::all)
        .setInputLayout(inputLayoutLightingPass)
        .setVertexLayout(tga::VertexLayout{sizeof(glm::vec2), {{0, tga::Format::r32g32_sfloat}}});
    tga::RenderPass lightingPass = tgai.createRenderPass(lightingPassInfo);
    #pragma endregion


    #pragma region 3rd renderPass initialization
    // create inputLayout for the third renderPass (post-proc)
    tga::InputLayout inputLayoutPostProcPass({{tga::BindingType::sampler}});  // Set = 0 : RenderedTex
    
    // create third renderPass : input intermediateResult (tex) -> output window
    tga::RenderPassInfo postProcPassInfo(vertexShaderTexToScreen, fragShaderPostProc, window);
    postProcPassInfo.setInputLayout(inputLayoutPostProcPass)
        .setVertexLayout(
            tga::VertexLayout{sizeof(glm::vec2), {{0, tga::Format::r32g32_sfloat}}});
    tga::RenderPass postProcPass = tgai.createRenderPass(postProcPassInfo);
    #pragma endregion 
     

    #pragma region create input sets
    
    #pragma region inputSets 1st
    // inputeSet camera for 1st pass
    tga::InputSet inputSetCamera_geometryPass = tgai.createInputSet({geometryPass, {tga::Binding{cameraData, 0}}, 0});

    // inputSets vertex buffer, index buffer, diffuse tex, # of indeces and # of instances for each model
    struct ModelRenderData {
        tga::Buffer vertexBuffer, indexBuffer;
        tga::InputSet gpuData;
        uint32_t indexCount;
        uint32_t numInstances;
    };
    // An std::unordered_map is a linked list, so traversal is relatively inefficient
    // For more efficient iteration during command buffer recording, store the necessary data in a compact vector
    std::vector<ModelRenderData> modelRenderData;
    modelRenderData.reserve(modelData.size());
    for (auto& [modelName, data] : modelData) {
        modelRenderData.push_back(
            {data.vertexBuffer, data.indexBuffer,
             tgai.createInputSet(
                 {geometryPass, {tga::Binding{data.modelMatrices, 0}, tga::Binding{data.colorTex, 1}}, 1}),
             data.indexCount, data.cfg.amount});
    }
    #pragma endregion

    #pragma region inputSets 2nd
    // inputSet camera and light for the 2nd pass
    tga::InputSet inputSetCameraLight_lightingPass =
        tgai.createInputSet({lightingPass, {tga::Binding{cameraData, 0}, tga::Binding{lightsBuffer, 1}}, 0});

    // inputSet GBuffer for the 2nd pass
    tga::InputSet inputSetGBuffer_lightingPass =
        tgai.createInputSet({lightingPass,
                             {tga::Binding{fragWorldPositions, 0}, tga::Binding{normals, 1}, tga::Binding{albedo, 2}},
                             1});
    #pragma endregion

    #pragma region inputSets 3rd
    // inputSet intermediate result tex for the 3rd pass
    tga::InputSet inputSetIntermediateRes_postProcPass =
        tgai.createInputSet({postProcPass, {tga::Binding{intermediateResult, 0}}, 0});
    #pragma endregion

    #pragma endregion
    

    #pragma region rendering loop
    //instantiate a commandBuffer
    std::vector<tga::CommandBuffer> cmdBuffers(tgai.backbufferCount(window));
    tga::CommandBuffer cmdBuffer{};
    
    //initialize timestamp to get deltaTime
    auto prevTime = std::chrono::steady_clock::now();
    // rendering loop
    while (!tgai.windowShouldClose(window)) {
        //compute deltaTime (dt)
        auto time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(time - prevTime).count();
        prevTime = time;


        // handle to the frameBuffer of the window
        uint32_t nextFrame = tgai.nextFrame(window);
        tga::CommandBuffer& cmdBuffer = cmdBuffers[nextFrame];
        if (!cmdBuffer) {
            // initialize a commandRecorder to start recording commands
            tga::CommandRecorder cmdRecorder = tga::CommandRecorder{tgai, cmdBuffer};

            // Upload the camera data and make sure the upload is finished before starting the vertex shader
            cmdRecorder.bufferUpload(camController->Data(), cameraData, sizeof(Camera))
                .barrier(tga::PipelineStage::Transfer, tga::PipelineStage::VertexShader);
                
           
            // 1. Geometry Pass
            cmdRecorder.setRenderPass(geometryPass, 0)
                .bindInputSet(inputSetCamera_geometryPass);

            for (auto& data : modelRenderData) {
                cmdRecorder.bindVertexBuffer(data.vertexBuffer)
                    .bindIndexBuffer(data.indexBuffer)
                    .bindInputSet(data.gpuData)
                    .drawIndexed(data.indexCount, 0, 0, data.numInstances);
            }

            // 2. Lighting Pass
            cmdRecorder.setRenderPass(lightingPass, 0)
                .bindInputSet(inputSetCameraLight_lightingPass)
                .bindInputSet(inputSetGBuffer_lightingPass)
                .bindVertexBuffer(vertexBuffer_quad)
                .bindIndexBuffer(indexBuffer_quad)
                .drawIndexed(6, 0, 0);

            // 3. PostProcessing Pass
            cmdRecorder.setRenderPass(postProcPass, nextFrame)
                .bindInputSet(inputSetIntermediateRes_postProcPass)
                .bindVertexBuffer(vertexBuffer_quad)
                .bindIndexBuffer(indexBuffer_quad)
                .drawIndexed(6, 0, 0);

            // the command recorder has done recording and can initialize a commandBuffer
            cmdBuffer = cmdRecorder.endRecording();

        } else {
            // Need to reset the command buffer before re-using it
            tgai.waitForCompletion(cmdBuffer);
        }
        //update camera data
        camController->update(dt);

        // execute the commands recorded in the commandBuffer
        tgai.execute(cmdBuffer);
        // present the current data in the frameBuffer "nextFrame" to the window
        tgai.present(window, nextFrame);
    }
    #pragma endregion

    return 0;
}
