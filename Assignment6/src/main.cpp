#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <random>

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
    tga::TextureBundle colorTex;
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
    /*auto windowWidth = static_cast<uint32_t>(8);
    auto windowHeight = static_cast<uint32_t>(8);*/
    tga::Window window = tgai.createWindow({windowWidth, windowHeight});
#pragma endregion

#pragma region load shaders
    const std::string vertexShader_TexToScreen_Path = "../shaders/renderTextureToScreen_vert.spv";
    const std::string fragShader_TexToScreen_Path = "../shaders/renderTextureToScreen_frag.spv";
    tga::Shader vertexShaderTexToScreen = tga::loadShader(vertexShader_TexToScreen_Path, tga::ShaderType::vertex, tgai);
    tga::Shader fragShaderTexToScreen = tga::loadShader(fragShader_TexToScreen_Path, tga::ShaderType::fragment, tgai);

    const std::string compShader_monteCarlo = "../shaders/monteCarlo_comp.spv";
    tga::Shader compShaderMonteCarlo = tga::loadShader(compShader_monteCarlo, tga::ShaderType::compute, tgai);
#pragma endregion

#pragma region load model data 
    //assuming only one mesh for simplicity
    tga::Obj obj = tga::loadObj("../../../Data/man/man.obj"); //using "man.obj"
    tga::Buffer vertexBuffer = makeBufferFromVector(tgai, tga::BufferUsage::vertex, obj.vertexBuffer);
    tga::Buffer indexBuffer = makeBufferFromVector(tgai, tga::BufferUsage::index, obj.indexBuffer);
    const std::string diffuseTexRelPath = "../../../Data/man/man_diffuse.png";
    tga::TextureBundle diffuseTex = tga::loadTexture(diffuseTexRelPath, tga::Format::r8g8b8a8_srgb, tga::SamplerMode::nearest, tgai);
#pragma endregion

#pragma region initialize model transforms
    glm::vec3 worldPos_man = glm::vec3(3.0f, 0.0f, 3.0f);
    Transform objTransform_man;
    objTransform_man.modelMatrix = glm::translate(glm::mat4(1.0f), worldPos_man) * glm::scale(glm::mat4(1), glm::vec3(0.02));
#pragma endregion

#pragma region initialize modelData
    ModelData modelData = {vertexBuffer, indexBuffer, objTransform_man, diffuseTex};
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
    Light light(glm::vec3(3.f, 5.f, 2.f), glm::vec4(1)); 
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
        {// Set = 0: Image data, Camera data, Random Tex
         {tga::BindingType::storageImage, tga::BindingType::uniformBuffer, tga::BindingType::storageImage},
         // Set = 1 : Vertex List, Index List
         {tga::BindingType::storageBuffer, tga::BindingType::storageBuffer}
        });
   
    tga::ComputePass computePass = tgai.createComputePass({compShaderMonteCarlo, inputLayoutComputePass});
#pragma endregion

#pragma region renderPass initialization
    tga::InputLayout inputLayoutRenderPass({// Set 0 : Texture
                                            {tga::BindingType::sampler}});

    tga::RenderPassInfo renderPassInfo(vertexShaderTexToScreen, fragShaderTexToScreen, window);
    renderPassInfo.setInputLayout(inputLayoutRenderPass)
        .setVertexLayout(tga::VertexLayout{sizeof(glm::vec2), {{0, tga::Format::r32g32_sfloat}}});
    
    tga::RenderPass renderPass = tgai.createRenderPass(renderPassInfo);

#pragma endregion

#pragma region inputSets computePass
    uint32_t randTexSide = 8; 
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);  // Adjust the range as needed

    // Specify the size of your vector
    size_t vectorSize = randTexSide*randTexSide*4;  // Adjust the size as needed

    // Create a vector and fill it with random floats
    std::vector<float> randomFloats(vectorSize);
    for (size_t i = 0; i < vectorSize; ++i) {
        randomFloats[i] = dis(gen);
    }


    tga::StagingBuffer randomTexStaging =
        tgai.createStagingBuffer({vectorSize * sizeof(float), tga::memoryAccess(randomFloats)});

    tga::Texture randomTex = tgai.createTexture({randTexSide, randTexSide, tga::Format::r32g32b32a32_sfloat, tga::SamplerMode::nearest, tga::AddressMode::clampEdge, tga::TextureType::_2D, 0, randomTexStaging});
    tga::Texture outputTex = tgai.createTexture({randTexSide, randTexSide, tga::Format::r32g32b32a32_sfloat});

    tga::InputSet inputSetCameraTexturesComputePass =
        tgai.createInputSet({computePass, {tga::Binding{outputTex, 0}, tga::Binding{cameraData, 1}, tga::Binding{randomTex, 2}}, 0});

    std::vector<glm::vec3> vertices;
    for (int i = 0; i < obj.vertexBuffer.size(); ++i) {
        tga::Vertex currentVertex = obj.vertexBuffer[i]; 
        vertices.push_back(currentVertex.position);
    }

    tga::Buffer verticesBuffer = makeBufferFromVector(tgai, tga::BufferUsage::storage, vertices);

    std::vector<glm::uvec3> indices;
    for (int i = 0; i < obj.indexBuffer.size(); i+=3) {
        glm::uvec3 index = glm::uvec3(obj.indexBuffer[i], obj.indexBuffer[i+1], obj.indexBuffer[i+2]);
        indices.push_back(index);
    }

    tga::Buffer indicesBuffer = makeBufferFromVector(tgai, tga::BufferUsage::storage, indices);


    tga::InputSet inputSetVertexIndexComputePass =
        tgai.createInputSet({computePass, {tga::Binding{verticesBuffer, 0}, tga::Binding{indicesBuffer, 1}}, 1});
#pragma endregion
        
#pragma region inputSets renderPass
    //TODO: texture to pass should be the result of the monte carlo compute pass
    tga::InputSet inputSetRenderPass = tgai.createInputSet({renderPass, {tga::Binding{outputTex, 0}}, 0});
    

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
        cmdRecorder.setComputePass(computePass).bindInputSet(inputSetCameraTexturesComputePass)
            .bindInputSet(inputSetVertexIndexComputePass).
            dispatch(windowWidth, windowHeight, 1);



        cmdRecorder.setRenderPass(renderPass, nextFrame)
            .bindVertexBuffer(vertexBuffer_quad)
            .bindIndexBuffer(indexBuffer_quad)
            .bindInputSet(inputSetRenderPass)
            .drawIndexed(6, 0, 0);

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


