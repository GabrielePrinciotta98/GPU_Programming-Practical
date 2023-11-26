#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>

#include "CameraController.hpp"
#include "tga/tga.hpp"
#include "tga/tga_createInfo_structs.hpp"
#include "tga/tga_utils.hpp"
namespace fs = std::filesystem;

struct Light {
    alignas(16) glm::vec3 lightPos = glm::vec3(0);
    alignas(16) glm::vec4 lightColor = glm::vec4(0);
};
//
// struct Transform {
//    alignas(16) glm::mat4 transform = glm::mat4(1);  // model world pos (model matrix)
//};
struct BoundingBox {
    glm::vec3 min;
    glm::vec3 max;
};

struct ConfigData {
    glm::vec3 initialPosition{0, 0, 0};
    float xstep{0}, ystep{0}; 
    float scale{0};
    uint32_t maxInstancesPerRow{0};
    uint32_t amount{0};
};

struct Size {
    uint32_t size;
};

struct Transform {
    glm::mat4 modelMatrix;
};

// data for each model
struct ModelData {
    std::vector<tga::Vertex> vertexList;
    std::vector<uint32_t> indexList;
    ConfigData cfg;
    tga::StagingBuffer staging_modelMatrices;
    tga::Buffer modelMatrices;
    tga::Buffer vertexBuffer, indexBuffer;
    uint32_t indexCount{0};
    tga::Texture colorTex;
    tga::Buffer bbData; //buffer holding reference to bounding box of this mesh
    tga::Buffer size; //buffer holding number of instances of this mesh, used as size for the computePass problem 
    //(in the lopp BufferDownload(visibilityBuffer, visibilityStaging)
    tga::StagingBuffer visibilityStaging; //staging buffer to get the output of the computeShader
    tga::Buffer visibilityBuffer; //buffer holding array of 0/1 flags  
};


// data for each model used in rendering loop 
struct ModelRenderData {
    std::string meshName;
    tga::Buffer vertexBuffer, indexBuffer;
    tga::StagingBuffer staging_modelMatrices;
    tga::Buffer modelMatrices;
    tga::InputSet computeInputSet; //inputSet for compute pass
    tga::InputSet geometryInputSet; //inputSet for geometry pass (1st pass)
    uint32_t indexCount;
    uint32_t numInstances;
    tga::StagingBuffer visibilityStaging;  // staging buffer to get the output of the computeShader
    tga::Buffer visibilityBuffer;          // buffer holding array of 0/1 flags
};

struct InstanceRenderData {
    tga::Buffer vertexBuffer, indexBuffer;
    tga::InputSet geometryInputSet;  // inputSet for geometry pass (1st pass)
    tga::StagingBuffer staging_modelMatrix;
    tga::Buffer modelMatrix;
};


// A little helper function to create a staging buffer that acts like a specific type
template <typename T>
std::tuple<T&, tga::StagingBuffer, size_t> stagingBufferOfType(tga::Interface& tgai)
{
    auto stagingBuff = tgai.createStagingBuffer({sizeof(T)});
    return {*static_cast<T *>(tgai.getMapping(stagingBuff)), stagingBuff, sizeof(T)};
}

template <typename T>
tga::Buffer makeBufferFromStruct(tga::Interface& tgai, tga::BufferUsage usage, T& data)
{
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

void rotateInstances(std::vector<ModelRenderData>& modelRenderData, tga::Interface& tgai);

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
    tga::Shader vertexShaderSecondPass =
        tga::loadShader(vertexShader_defRendering2_Path, tga::ShaderType::vertex, tgai);
    tga::Shader fragShaderSecondPass = tga::loadShader(fragShader_defRendering2_Path, tga::ShaderType::fragment, tgai);

    const std::string vertexShader_TexToScreen_Path = "../shaders/renderTextureToScreen_vert.spv";
    const std::string fragShader_PostProcessing_Path = "../shaders/postProcessing_frag.spv";
    tga::Shader vertexShaderTexToScreen = tga::loadShader(vertexShader_TexToScreen_Path, tga::ShaderType::vertex, tgai);
    tga::Shader fragShaderPostProc = tga::loadShader(fragShader_PostProcessing_Path, tga::ShaderType::fragment, tgai);

    const std::string compShader_frustumCulling = "../shaders/frustumCulling_comp.spv";
    tga::Shader compShaderFrustumCulling = tga::loadShader(compShader_frustumCulling, tga::ShaderType::compute, tgai);
#pragma endregion

#pragma region load model data
    std::unordered_map<std::filesystem::path, ModelData> modelData;

    // std::cout << std::filesystem::current_path() << std::endl;
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
            xstep
            ystep
            scale
            maxInstancesPerRow
            instanceAmount
             *
             Delimited by white space
             */
            auto& cfg = modelData[entry.path().stem()].cfg;
            config >> cfg.initialPosition.x >> cfg.initialPosition.y >> cfg.initialPosition.z;
            config >> cfg.xstep;
            config >> cfg.ystep;
            config >> cfg.scale;
            config >> cfg.maxInstancesPerRow;
            config >> cfg.amount;
            Size s{cfg.amount};
            modelData[entry.path().stem()].size = makeBufferFromStruct(tgai, tga::BufferUsage::uniform, s);
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

            data.vertexList = obj.vertexBuffer;
            data.indexList = obj.indexBuffer;
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

#pragma region assigning an object space BB to each mesh
    //std::vector<BoundingBox> boundingBoxes;

    for (auto& [modelName, data] : modelData) {
        glm::vec3 min{INFINITY, INFINITY, INFINITY};
        glm::vec3 max{-INFINITY, -INFINITY, -INFINITY};

        for (int i = 0; i < data.vertexList.size(); ++i) {
            glm::vec3 vertex = data.vertexList[i].position;

            min = glm::min(min, vertex);
            max = glm::min(max, vertex);
        }

        BoundingBox bb{min, max};
        data.bbData = makeBufferFromStruct(tgai, tga::BufferUsage::uniform, bb);
        //boundingBoxes.push_back(data.bb);
    }

    //tga::Buffer boundingBoxesBuffer = makeBufferFromVector(tgai, tga::BufferUsage::storage, boundingBoxes);
#pragma endregion

    for (auto& [modelName, data] : modelData) {
        size_t visibilityBufferSize = data.cfg.amount * sizeof(uint32_t);
        data.visibilityBuffer = tgai.createBuffer({tga::BufferUsage::storage, visibilityBufferSize});
        data.visibilityStaging = tgai.createStagingBuffer({visibilityBufferSize});

        auto visibility = static_cast<uint32_t *>(tgai.getMapping(data.visibilityStaging));
        for (uint32_t i = 0; i < data.cfg.amount; ++i) {
            visibility[i] = 0; //initialize all instances as not visible
        }
    }

#pragma region create storage buffers for the model matrices of the instances for each model

    for (auto& [modelName, data] : modelData) {
        auto matrixStaging = tgai.createStagingBuffer({sizeof(glm::mat4) * data.cfg.amount});
        std::span<glm::mat4> matrixData{static_cast<glm::mat4 *>(tgai.getMapping(matrixStaging)), data.cfg.amount};

        float xStep = data.cfg.xstep;
        float yStep = data.cfg.ystep;
        glm::vec3 initialPosition = data.cfg.initialPosition;
        size_t maxInstancesPerRow = data.cfg.maxInstancesPerRow;

        size_t rowCount = static_cast<size_t>(std::ceil(static_cast<float>(data.cfg.amount) / maxInstancesPerRow));

        for (size_t row = 0; row < rowCount; ++row) {
            size_t instancesInThisRow = std::min(maxInstancesPerRow, data.cfg.amount - row * maxInstancesPerRow);

            for (size_t col = 0; col < instancesInThisRow; ++col) {
                auto& matrix = matrixData[row * maxInstancesPerRow + col];

                // Calculate position based on row, column, and steps
                glm::vec3 position = initialPosition + glm::vec3(col * xStep, 0.0f, row * yStep);

                matrix = glm::translate(glm::mat4(1), position) * glm::scale(glm::mat4(1), glm::vec3(data.cfg.scale));
            }
        }

        data.staging_modelMatrices = matrixStaging;
        data.modelMatrices = tgai.createBuffer({tga::BufferUsage::storage, matrixData.size_bytes(), matrixStaging});
    }

#pragma endregion



#pragma region initialize camera controller and create camera buffer
    const glm::vec3 startPosition = glm::vec3(0.f, 2.f, 0.f);
    float aspectRatio = windowWidth / static_cast<float>(windowHeight);
    std::unique_ptr<CameraController> camController = std::make_unique<CameraController>(
        tgai, window, 60, aspectRatio, 0.1f, 30000.f, startPosition, glm::vec3{0, 0, 1}, glm::vec3{0, 1, 0});
    tga::Buffer cameraData =
        tgai.createBuffer(tga::BufferInfo{tga::BufferUsage::uniform, sizeof(Camera), camController->Data()});
#pragma endregion

#pragma region initialize lights, assigning different positions and colors
    std::vector<Light> lights;
    int nLigths = 144;
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            lights.push_back({glm::vec3(30, 30 - 2 * i, 5 - 2 * j),
                              glm::vec4(glm::vec3(0.01 + 0.002 * i, 0.01 + 0.002 * j, 0.005 + i * 0.001), 1)});
        }
    }

    tga::Buffer lightsBuffer = makeBufferFromVector(tgai, tga::BufferUsage::uniform, lights);
#pragma endregion

#pragma region create vBuffer and iBuffer for quad needed for render to tex
    // Data for a quad (quad vertex buffer)
    std::vector<glm::vec2> quadVertices = {glm::vec2(-1, -1), glm::vec2(1, 1), glm::vec2(-1, 1), glm::vec2(1, -1)};
    tga::Buffer vertexBuffer_quad = makeBufferFromVector(tgai, tga::BufferUsage::vertex, quadVertices);

    // Indices for a quad, aka 2 triangles (quad index buffer)
    std::vector<uint32_t> quadIndices = {0, 1, 2, 0, 3, 1};
    tga::Buffer indexBuffer_quad = makeBufferFromVector(tgai, tga::BufferUsage::index, quadIndices);
#pragma endregion

#pragma region computePass initialization
    tga::InputLayout inputLayoutComputePass({
        // Set = 0: Camera data
        {tga::BindingType::uniformBuffer},
        // Set = 1: Transform data, Bounding Box data, Size data (# of instances of current mesh), visibilityBuffer
        {tga::BindingType::storageBuffer, tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer, tga::BindingType::storageBuffer},
    });
   
    tga::ComputePass computePass = tgai.createComputePass({compShaderFrustumCulling, inputLayoutComputePass});
#pragma endregion

#pragma region 1st renderPass initialization
    // create inputLayout for the first renderPass
    tga::InputLayout inputLayoutGeometryPass({// Set = 0: Camera data
                                              {tga::BindingType::uniformBuffer},
                                              // Set = 1: Transform data, Diffuse Tex
                                              {tga::BindingType::storageBuffer, tga::BindingType::sampler, tga::BindingType::storageBuffer}});

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
    tga::InputLayout inputLayoutLightingPass(
        {// Set = 0: Camera data, Light data
         {tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer},
         // Set = 1 : fragWorldpos, normals, albedo
         {tga::BindingType::sampler, tga::BindingType::sampler, tga::BindingType::sampler}});

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
        .setVertexLayout(tga::VertexLayout{sizeof(glm::vec2), {{0, tga::Format::r32g32_sfloat}}});
    tga::RenderPass postProcPass = tgai.createRenderPass(postProcPassInfo);
#pragma endregion

#pragma region create input sets

#pragma region inputSets computePass
    // inputeSet camera for compute pass
    tga::InputSet inputSetCamera_computePass = tgai.createInputSet({computePass, {tga::Binding{cameraData, 0}}, 0});
    

    // inputSets vertex buffer, index buffer, diffuse tex, # of indeces and # of instances for each model
    
    // An std::unordered_map is a linked list, so traversal is relatively inefficient
    // For more efficient iteration during command buffer recording, store the necessary data in a compact vector
    std::vector<ModelRenderData> modelRenderData;
    modelRenderData.reserve(modelData.size());
    for (auto& [modelName, data] : modelData) {
        modelRenderData.push_back(
            {modelName.string(), data.vertexBuffer, data.indexBuffer, data.staging_modelMatrices, data.modelMatrices,
             tgai.createInputSet({computePass,
                                  {tga::Binding{data.modelMatrices, 0}, tga::Binding{data.bbData, 1},
                                   tga::Binding{data.size, 2}, tga::Binding{data.visibilityBuffer, 3}},
                                  1}),
             tgai.createInputSet({geometryPass,
                                  {tga::Binding{data.modelMatrices, 0}, tga::Binding{data.colorTex, 1}, tga::Binding{data.visibilityBuffer, 2}},
                                  1}),
             data.indexCount, data.cfg.amount, data.visibilityStaging, data.visibilityBuffer});
    }
#pragma endregion

    // inputeSet camera for 1st pass
    tga::InputSet inputSetCamera_geometryPass = tgai.createInputSet({geometryPass, {tga::Binding{cameraData, 0}}, 0});

    std::vector<InstanceRenderData> instanceRenderData;
    for (auto& [modelName, data] : modelData) {
        std::span<Transform> matrixData{static_cast<Transform *>(tgai.getMapping(data.staging_modelMatrices)), data.cfg.amount};
        for (int i = 0; i < data.cfg.amount; ++i) {
            tga::StagingBuffer currentModelMatrixStaging =
                tgai.createStagingBuffer({sizeof(Transform), tga::memoryAccess(matrixData[i])});
            tga::Buffer currentModelMatrixBuffer =
                tgai.createBuffer({tga::BufferUsage::storage, sizeof(Transform), currentModelMatrixStaging
        });


            instanceRenderData.push_back({
                data.vertexBuffer,
                data.indexBuffer,
                tgai.createInputSet(
                    {geometryPass, {tga::Binding{currentModelMatrixBuffer, 0}, tga::Binding{data.colorTex, 1}}, 1}),
                currentModelMatrixStaging, currentModelMatrixBuffer
            });
        }
    }

#pragma region inputSets 2nd
    // inputSet camera and light for the 2nd pass
    tga::InputSet inputSetCameraLight_lightingPass =
        tgai.createInputSet({lightingPass, {tga::Binding{cameraData, 0}, tga::Binding{lightsBuffer, 1}}, 0});

    // inputSet GBuffer for the 2nd pass
    tga::InputSet inputSetGBuffer_lightingPass = tgai.createInputSet(
        {lightingPass, {tga::Binding{fragWorldPositions, 0}, tga::Binding{normals, 1}, tga::Binding{albedo, 2}}, 1});
#pragma endregion

#pragma region inputSets 3rd
    // inputSet intermediate result tex for the 3rd pass
    tga::InputSet inputSetIntermediateRes_postProcPass =
        tgai.createInputSet({postProcPass, {tga::Binding{intermediateResult, 0}}, 0});
#pragma endregion

#pragma endregion

#pragma region rendering loop
    // instantiate a commandBuffer  
    tga::CommandBuffer cmdBuffer{};

    // initialize timestamp to get deltaTime
    auto prevTime = std::chrono::steady_clock::now();
    // rendering loop
    while (!tgai.windowShouldClose(window)) {
        // compute deltaTime (dt)
        auto time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(time - prevTime).count();
        prevTime = time;

        // handle to the frameBuffer of the window
        uint32_t nextFrame = tgai.nextFrame(window);
     
        // initialize a commandRecorder to start recording commands
        tga::CommandRecorder cmdRecorder = tga::CommandRecorder{tgai, cmdBuffer};


        // Upload the updated camera data and make sure the upload is finished before starting the vertex shader
        cmdRecorder.bufferUpload(camController->Data(), cameraData, sizeof(Camera))
            .barrier(tga::PipelineStage::Transfer, tga::PipelineStage::VertexShader);

        // Upload the updated model matrices data and make sure the upload is finished before starting the vertex shader
        for (auto& data : modelRenderData) {
            auto matrixStaging = data.staging_modelMatrices;
            auto matrixBuffer = data.modelMatrices;
            cmdRecorder.bufferUpload(matrixStaging, matrixBuffer, sizeof(glm::mat4) * data.numInstances)
                .barrier(tga::PipelineStage::Transfer, tga::PipelineStage::VertexShader);
        }

        // Upload the updated model matrices data and make sure the upload is finished before starting the vertex shader
        for (auto& data : instanceRenderData) {
            auto matrixStaging = data.staging_modelMatrix;
            auto matrixBuffer = data.modelMatrix;
            cmdRecorder.bufferUpload(matrixStaging, matrixBuffer, sizeof(Transform))
                .barrier(tga::PipelineStage::Transfer, tga::PipelineStage::VertexShader);
        }


        //TODO: COMPUTE PASS
        constexpr auto workGroupSize = 64;
        for (auto& data : modelRenderData) {
            cmdRecorder.setComputePass(computePass)
                .bindInputSet(inputSetCamera_computePass)
                .bindInputSet(data.computeInputSet)
                .dispatch((data.numInstances + (workGroupSize - 1)) / workGroupSize, 1, 1)
                .barrier(tga::PipelineStage::ComputeShader, tga::PipelineStage::Transfer)
                .bufferDownload(data.visibilityBuffer, data.visibilityStaging, data.numInstances * sizeof(uint32_t))
                .barrier(tga::PipelineStage::ComputeShader, tga::PipelineStage::VertexShader);
        }



#pragma region initialize draw indirect commands with different amount of instances (visible objects) every loop
        std::vector<tga::DrawIndexedIndirectCommand> indirectCommands;
        std::string title; 
        for (auto& data : modelRenderData) {
            uint32_t visibleInstances{0}; 
            auto visibility = static_cast<uint32_t *>(tgai.getMapping(data.visibilityStaging));
            for (uint32_t i = 0; i < data.numInstances; ++i) {
                visibleInstances += visibility[i];  // get the number of visible instances;
                //indirectCommands.push_back({data.indexCount, visibility[i], 0, 0, 0});
                //std::cout << visibility[i] << std::endl;
            }
            //std::cout << data.meshName + ": " + std::to_string(visibleInstances) << std::endl;
            title += data.meshName + ": " + std::to_string(visibleInstances) + ",   ";  
            indirectCommands.push_back({data.indexCount, data.numInstances, 0, 0, 0});
        }
        tgai.setWindowTitle(window, title);

        tga::Buffer indirectDrawBuffer = makeBufferFromVector(tgai, tga::BufferUsage::indirect, indirectCommands);
#pragma endregion

        // 1. Geometry Pass
        cmdRecorder.setRenderPass(geometryPass, 0).bindInputSet(inputSetCamera_geometryPass);


        for (auto& data : modelRenderData) {
            cmdRecorder.bindVertexBuffer(data.vertexBuffer)
                .bindIndexBuffer(data.indexBuffer)
                .bindInputSet(data.geometryInputSet)
                .drawIndexedIndirect(indirectDrawBuffer, indirectCommands.size());
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

        
        // execute the commands recorded in the commandBuffer
        tgai.execute(cmdBuffer);
        // present the current data in the frameBuffer "nextFrame" to the window
        tgai.present(window, nextFrame);


        #pragma region update data every frame 
        // update camera data
        camController->update(dt);
        // update model matrices
        rotateInstances(modelRenderData, tgai);
        #pragma endregion

    }
#pragma endregion

    return 0;
}

void rotateInstances(std::vector<ModelRenderData>& modelRenderData, tga::Interface& tgai)
{
    for (auto& data : modelRenderData) {
        auto matrixStaging = data.staging_modelMatrices;
        std::span<glm::mat4> matrixData{static_cast<glm::mat4 *>(tgai.getMapping(matrixStaging)), data.numInstances};
        for (size_t i = 0; i < matrixData.size(); ++i) {
            auto& matrix = matrixData[i];
            float angle = glm::radians(1.0f);
            matrix = matrix * glm::rotate(glm::mat4(1), angle, glm::vec3(0., 1., 0.));
        }
        data.staging_modelMatrices = matrixStaging;
    }
}
