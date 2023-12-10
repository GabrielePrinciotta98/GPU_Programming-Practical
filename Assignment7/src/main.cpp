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

struct CameraData {
    glm::mat4 view;
    glm::mat4 toWorld;
    glm::mat4 projection;
    glm::mat4 invProjection;
};

struct VertexAttribute {
    glm::vec3 normal;
    glm::vec2 texCoords;
};

struct Triangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

struct AABB {
    glm::vec4 min; //using vec4 for alignment reasons
    glm::vec4 max; //using vec4 for alignment reasons
};

struct BVHNodeData {
    int left;   // Index of the left child in the nodes array, or -1 if leaf
    int right;  // Index of the right child in the nodes array, or -1 if leaf
    int first;  // First index of the primitive in this node (if leaf)
    int count;  // Number of primitives in this node (if leaf)
};

struct BVHNode {
    AABB bounds;
    BVHNodeData data;
};

struct BVHRootNodeIndex {
    uint32_t root;
};

AABB computeBounds(const std::vector<Triangle>& triangles, int start, int end)
{
    glm::vec4 min = glm::vec4(std::numeric_limits<float>::max());
    glm::vec4 max = glm::vec4(-std::numeric_limits<float>::max());

    for (int i = start; i < end; ++i) {
        const Triangle& tri = triangles[i];
        min = glm::vec4(glm::min(glm::vec3(min), glm::min(tri.v0, glm::min(tri.v1, tri.v2))), 1);
        max = glm::vec4(glm::max(glm::vec3(max), glm::max(tri.v0, glm::max(tri.v1, tri.v2))), 1);
    }

    return {min, max};
}


int chooseSplitPosition(const std::vector<Triangle>& triangles, int start, int end)
{
    // Simply split the range in half
    return start + (end - start) / 2;
}

int buildBVH(std::vector<BVHNode>& nodes, std::vector<Triangle>& triangles, int start, int end)
{
    if (end - start == 1) {
        nodes.push_back({computeBounds(triangles, start, end), -1, -1, start, end - start});
        return nodes.size() - 1;
    }

    AABB bounds = computeBounds(triangles, start, end);
    int split = chooseSplitPosition(triangles, start, end);

    int left = buildBVH(nodes, triangles, start, split);
    int right = buildBVH(nodes, triangles, split, end);

    nodes.push_back({bounds, left, right, -1, -1});
    return nodes.size() - 1;
}



int main()
{
    std::cout << "GPU Pro\n";
    tga::Interface tgai;

#pragma region create window
    uint32_t resolutionScale{40};  // 80: 720p, 120: 1080p
    uint32_t windowWidth = 16 * resolutionScale, windowHeight = 9 * resolutionScale;

    auto window = tgai.createWindow({windowWidth, windowHeight, tga::PresentMode::immediate});
#pragma endregion

#pragma region load shaders
    const std::string vertexShader_TexToScreen_Path = "../shaders/renderTextureToScreen_vert.spv";
    const std::string fragShader_TexToScreen_Path = "../shaders/renderTextureToScreen_frag.spv";
    tga::Shader vs = tga::loadShader(vertexShader_TexToScreen_Path, tga::ShaderType::vertex, tgai);
    tga::Shader fs = tga::loadShader(fragShader_TexToScreen_Path, tga::ShaderType::fragment, tgai);

    const std::string compShader_monteCarlo = "../shaders/monteCarlo_comp.spv";
    tga::Shader cs = tga::loadShader(compShader_monteCarlo, tga::ShaderType::compute, tgai);
#pragma endregion
        
#pragma region load model data 
    //assuming only one mesh for simplicity
    tga::Obj model = tga::loadObj("../../../Data/suzanne/suzanne.obj"); 
#pragma endregion

#pragma region populate vertexBuffer, vertexAttributes and indexBuffer 
    std::vector<float> vertexBufferCPU;
    std::vector<float> vertexAttributes;

    vertexBufferCPU.reserve(model.vertexBuffer.size());
    vertexAttributes.reserve(model.vertexBuffer.size());
    for (auto& v : model.vertexBuffer) {
        vertexBufferCPU.push_back(v.position.x);
        vertexBufferCPU.push_back(v.position.y);
        vertexBufferCPU.push_back(v.position.z);

        vertexAttributes.push_back(v.normal.x);
        vertexAttributes.push_back(v.normal.y);
        vertexAttributes.push_back(v.normal.z);
        vertexAttributes.push_back(v.uv.x);
        vertexAttributes.push_back(v.uv.y);
    }

    std::vector<uint32_t>& indexBufferCPU = model.indexBuffer;
#pragma endregion
    //create triangle array
    std::vector<Triangle> triangles;
    for (size_t i = 0; i < indexBufferCPU.size(); i += 3) {
        glm::vec3 v0(vertexBufferCPU[indexBufferCPU[i] * 3], vertexBufferCPU[indexBufferCPU[i] * 3 + 1],
                     vertexBufferCPU[indexBufferCPU[i] * 3 + 2]);

        glm::vec3 v1(vertexBufferCPU[indexBufferCPU[i + 1] * 3], vertexBufferCPU[indexBufferCPU[i + 1] * 3 + 1],
                     vertexBufferCPU[indexBufferCPU[i + 1] * 3 + 2]);

        glm::vec3 v2(vertexBufferCPU[indexBufferCPU[i + 2] * 3], vertexBufferCPU[indexBufferCPU[i + 2] * 3 + 1],
                     vertexBufferCPU[indexBufferCPU[i + 2] * 3 + 2]);

        triangles.push_back({v0, v1, v2});
    }

    // Build BVH
    static_assert(alignof(glm::vec3) == 16);
    std::vector<BVHNode> bvhNodes;
    BVHRootNodeIndex rootNodeIndex(buildBVH(bvhNodes, triangles, 0, triangles.size()));
    std::cout << rootNodeIndex.root << std::endl;
    tga::StagingBuffer rootNodeStaging =
        tgai.createStagingBuffer({sizeof(BVHRootNodeIndex), tga::memoryAccess(rootNodeIndex)});
    tga::Buffer rootNodeBuffer = tgai.createBuffer({tga::BufferUsage::uniform, sizeof(BVHRootNodeIndex), rootNodeStaging});

  
    std::vector<tga::DrawIndexedIndirectCommand> drawCmds{{.indexCount = static_cast<uint32_t>(indexBufferCPU.size()),
                                                           .instanceCount = 1,
                                                           .firstIndex = 0,
                                                           .vertexOffset = 0,
                                                           .firstInstance = 0}};


    auto makeStorageBuffer = [&](auto& vec) {
        auto size = sizeof(vec[0]) * vec.size();
        auto staging = tgai.createStagingBuffer({size, tga::memoryAccess(vec)});
        auto buf = tgai.createBuffer({tga::BufferUsage::storage, size, staging});
        tgai.free(staging);
        return buf;
    };


    auto vertexBuffer = makeStorageBuffer(vertexBufferCPU);
    auto indexBuffer = makeStorageBuffer(indexBufferCPU);
    auto vertexAttributeBuffer = makeStorageBuffer(vertexAttributes);
    auto drawCmdBuffer = makeStorageBuffer(drawCmds);
    auto trianglesBuffer = makeStorageBuffer(triangles);
    auto bvhNodesBuffer = makeStorageBuffer(bvhNodes);


    auto texFormat = tga::Format::r16g16b16a16_sfloat;
    // May interpolate between color data
    auto renderTex0 = tgai.createTexture({windowWidth, windowHeight, texFormat, tga::SamplerMode::linear});
    auto renderTex1 = tgai.createTexture({windowWidth, windowHeight, texFormat, tga::SamplerMode::linear});
    // Don't interpolate between position data
    auto positionTex0 = tgai.createTexture({windowWidth, windowHeight, texFormat, tga::SamplerMode::nearest});
    auto positionTex1 = tgai.createTexture({windowWidth, windowHeight, texFormat, tga::SamplerMode::nearest});

    // Ray tracing pass
    auto cp = tgai.createComputePass(
        {cs, tga::InputLayout{tga::SetLayout{tga::BindingType::uniformBuffer, tga::BindingType::storageBuffer,
                                             tga::BindingType::storageBuffer, tga::BindingType::storageBuffer,
                                             tga::BindingType::storageBuffer},
                              tga::SetLayout{tga::BindingType::storageImage, tga::BindingType::sampler,
                                             tga::BindingType::uniformBuffer, tga::BindingType::uniformBuffer,
                                             tga::BindingType::storageImage, tga::BindingType::sampler},
                              tga::SetLayout{tga::BindingType::storageBuffer, tga::BindingType::storageBuffer, 
                                             tga::BindingType::uniformBuffer}
                            }});

    auto sceneDataBuffer = tgai.createBuffer({tga::BufferUsage::uniform, sizeof(uint32_t)});
    // Double buffer the camera
    auto camBuffer0 = tgai.createBuffer({tga::BufferUsage::uniform, sizeof(CameraData)});
    auto camBuffer1 = tgai.createBuffer({tga::BufferUsage::uniform, sizeof(CameraData)});


    // Presentation pass
    auto rp = tgai.createRenderPass(tga::RenderPassInfo{vs, fs, window}.setInputLayout(
        tga::InputLayout{tga::SetLayout{tga::BindingType::storageImage}}));
    std::vector<tga::CommandBuffer> cmdBuffers(tgai.backbufferCount(window));


    // Constant data
    auto computeInput = tgai.createInputSet(
        {cp,
         {tga::Binding{sceneDataBuffer, 0}, tga::Binding{vertexBuffer, 1}, tga::Binding{indexBuffer, 2},
          tga::Binding{drawCmdBuffer, 3}, tga::Binding{vertexAttributeBuffer, 4}},
         0});

    // Double buffer the texture and the camera input
    auto computeRender0 = tgai.createInputSet(
        {cp,
         {tga::Binding{renderTex0, 0}, tga::Binding{renderTex1, 1}, tga::Binding{camBuffer0, 2},
          tga::Binding{camBuffer1, 3}, tga::Binding{positionTex0, 4}, tga::Binding{positionTex1, 5}},
         1});
    auto computeRender1 = tgai.createInputSet(
        {cp,
         {tga::Binding{renderTex1, 0}, tga::Binding{renderTex0, 1}, tga::Binding{camBuffer1, 2},
          tga::Binding{camBuffer0, 3}, tga::Binding{positionTex1, 4}, tga::Binding{positionTex0, 5}},
         1});
    
    //BVH related input
    auto computeInput2 = tgai.createInputSet(
        {cp, {tga::Binding{bvhNodesBuffer, 0}, tga::Binding{trianglesBuffer, 1}, tga::Binding{rootNodeBuffer, 2}},
            2});

    // Which texture to present
    auto presentInput0 = tgai.createInputSet({rp, {tga::Binding{renderTex0, 0}}, 0});
    auto presentInput1 = tgai.createInputSet({rp, {tga::Binding{renderTex1, 0}}, 0});

#pragma region camera setting
    float fov{70}, aspectRatio{windowWidth / float(windowHeight)}, nearPlane{0.1}, farPlane{1000.};
    CameraData cam, prevCam;
    float pitch{0}, yaw{140};
    float speed = 7.;
    float speedBoost = 8;
    float turnSpeed = 75;
    glm::vec3 camPos{-5, 0.5, 5};

    auto moveCamera = [&](float dt) {
        float moveSpeed = speed;
        if (tgai.keyDown(window, tga::Key::R)) moveSpeed *= speedBoost;
        if (tgai.keyDown(window, tga::Key::Left)) yaw += dt * turnSpeed;
        if (tgai.keyDown(window, tga::Key::Right)) yaw -= dt * turnSpeed;
        if (tgai.keyDown(window, tga::Key::Up)) pitch += dt * turnSpeed;
        if (tgai.keyDown(window, tga::Key::Down)) pitch -= dt * turnSpeed;
        pitch = std::clamp(pitch, -89.f, 89.f);
        auto rot = glm::mat3_cast(glm::quat(glm::vec3(-glm::radians(pitch), glm::radians(yaw), 0.f)));
        glm::vec3 lookDir = rot * glm::vec3{0, 0, 1};
        auto r = rot * glm::cross(glm::vec3(0, 1, 0), glm::vec3{0, 0, 1});

        if (tgai.keyDown(window, tga::Key::W)) camPos += lookDir * dt * moveSpeed;
        if (tgai.keyDown(window, tga::Key::S)) camPos -= lookDir * dt * moveSpeed;

        if (tgai.keyDown(window, tga::Key::A)) camPos += r * dt * moveSpeed;
        if (tgai.keyDown(window, tga::Key::D)) camPos -= r * dt * moveSpeed;

        if (tgai.keyDown(window, tga::Key::Space)) camPos += glm::vec3{0, 1, 0} * dt * moveSpeed;
        if (tgai.keyDown(window, tga::Key::Shift_Left)) camPos -= glm::vec3{0, 1, 0} * dt * moveSpeed;

        cam.projection = glm::perspective_vk(glm::radians(fov), aspectRatio, nearPlane, farPlane);
        cam.view = glm::lookAt(camPos, camPos + lookDir, glm::vec3{0, 1, 0});
        cam.toWorld = glm::inverse(cam.view);
        cam.invProjection = glm::inverse(cam.projection);
    };

    moveCamera(0);
#pragma endregion

    tga::CommandBuffer cmd;

    double smoothedTime{0};
    auto prevTime = std::chrono::steady_clock::now();
    auto frame0Staging = tgai.createStagingBuffer({sizeof(CameraData) * 2 + sizeof(uint32_t)});
    auto frame1Staging = tgai.createStagingBuffer({sizeof(CameraData) * 2 + sizeof(uint32_t)});

    auto frame0Data = tgai.getMapping(frame0Staging);
    auto frame1Data = tgai.getMapping(frame1Staging);

    for (uint32_t frame{0}; !tgai.windowShouldClose(window); ++frame) {
        auto nextFrame = tgai.nextFrame(window);
        auto& cmd = cmdBuffers[nextFrame];

        // Record a command Buffer only once per distinct window frame buffer
        bool f = (frame % 2 == 0);
        if (!cmd || cmdBuffers.size() != 2) {
            constexpr size_t WGS{8};
            cmd = tga::CommandRecorder{tgai, cmd}
                      // Upload the data, no staging buffer since the data is only a couple of bytes
                      .bufferUpload(f ? frame0Staging : frame1Staging, f ? camBuffer0 : camBuffer1, sizeof(CameraData))
                      .bufferUpload(f ? frame0Staging : frame1Staging, f ? camBuffer1 : camBuffer0, sizeof(CameraData),
                                    sizeof(CameraData))
                      .bufferUpload(f ? frame0Staging : frame1Staging, sceneDataBuffer, sizeof(uint32_t),
                                    sizeof(CameraData) * 2)
                      .barrier(tga::PipelineStage::Transfer, tga::PipelineStage::ComputeShader)
                      // Execute the compute pass for path tracing, switch which textures are written to and which are
                      // read from
                      .setComputePass(cp)
                      .bindInputSet(computeInput)
                      .bindInputSet(computeInput2)  
                      .bindInputSet(f ? computeRender0 : computeRender1)
                      .dispatch((windowWidth + WGS - 1) / WGS, (windowHeight + WGS - 1) / WGS, 1)
                      .barrier(tga::PipelineStage::ComputeShader, tga::PipelineStage::FragmentShader)
                      // Present the results. may lag one frame behind since we don't switch which one we read from
                      .setRenderPass(rp, nextFrame)
                      .bindInputSet(f ? presentInput0 : presentInput1)
                      .draw(3, 0)
                      .endRecording();
        } else {
            tgai.waitForCompletion(cmd);
        }

        auto time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(time - prevTime).count();
        if (frame % 256 == 0) {
            smoothedTime = 0;
        }
        smoothedTime += dt;
        prevTime = time;

        // Set approximate fps
        tgai.setWindowTitle(window, std::to_string((frame % 256) / smoothedTime));

        prevCam = cam;
        moveCamera(dt);
        // Write to the buffer where reading is completed
        auto writeTo = f ? frame0Data : frame1Data;

        // Update the staging buffer
        std::memcpy(writeTo, &cam, sizeof(cam));
        std::memcpy(static_cast<char *>(writeTo) + sizeof(CameraData), &prevCam, sizeof(CameraData));
        std::memcpy(static_cast<char *>(writeTo) + 2 * sizeof(CameraData), &frame, sizeof(uint32_t));

        tgai.execute(cmd);
        tgai.present(window, nextFrame);
    }



    return 0;
}


