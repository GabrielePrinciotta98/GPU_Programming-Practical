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


struct Transform {
    glm::mat4 modelMatrix = glm::mat4(1);
};

struct ModelData {
    tga::Buffer vertexBuffer;
    tga::Buffer indexBuffer;
    Transform modelTransform;
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

#pragma endregion

    

#pragma region create buffers for the model matrices of the instances for each model

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
    tga::InputLayout inputLayoutComputePass(
        {// Set = 0: Camera data
         {tga::BindingType::uniformBuffer},
         // Set = 1: Transform data, Bounding Box data, Size data (# of instances of current mesh), visibilityBuffer,
         // indirectDrawCommands, indexCount
         {tga::BindingType::storageBuffer, tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer,
          tga::BindingType::storageBuffer, tga::BindingType::storageBuffer, tga::BindingType::uniformBuffer}

        });
   
    tga::ComputePass computePass = tgai.createComputePass({compShaderFrustumCulling, inputLayoutComputePass});
#pragma endregion

#pragma region renderPass initialization

#pragma endregion

#pragma region inputSets computePass

#pragma endregion

#pragma region inputSets renderPass

    

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


        //// Upload the updated camera data and make sure the upload is finished before starting the vertex shader
        //cmdRecorder.bufferUpload(camController->Data(), cameraData, sizeof(Camera))
        //    .barrier(tga::PipelineStage::Transfer, tga::PipelineStage::VertexShader);

        

        // the command recorder has done recording and can initialize a commandBuffer
        cmdBuffer = cmdRecorder.endRecording();

        
        // execute the commands recorded in the commandBuffer
        tgai.execute(cmdBuffer);
        // present the current data in the frameBuffer "nextFrame" to the window
        tgai.present(window, nextFrame);


        #pragma region update data every frame 
        // update camera data
        //camController->update(dt);
        #pragma endregion

    }
#pragma endregion

    return 0;
}


