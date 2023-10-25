#include <iostream>

#include "tga/tga.hpp"
#include "tga/tga_createInfo_structs.hpp"
#include "tga/tga_utils.hpp"

int main()
{
    std::cout << "GPU Pro\n";
    tga::Interface tgai;

    const std::string vertexShaderRelPath = "../shaders/triangle_vert.spv";
    const std::string fragShaderRelPath = "../shaders/triangle_frag.spv";
    
    tga::Shader vertexShader = tga::loadShader(vertexShaderRelPath, tga::ShaderType::vertex, tgai);
    tga::Shader fragShader = tga::loadShader(fragShaderRelPath, tga::ShaderType::fragment, tgai);

    tga::Window window = tgai.createWindow({960, 540});

    tga::RenderPassInfo renderPassInfo(vertexShader, fragShader, window);
    tga::RenderPass renderPass = tgai.createRenderPass(renderPassInfo);

    tga::CommandBuffer cmdBuffer{};

    while (!tgai.windowShouldClose(window)) {
        uint32_t nextFrame = tgai.nextFrame(window);
        tga::CommandRecorder cmdRecorder = tga::CommandRecorder{tgai, cmdBuffer};
        cmdRecorder.setRenderPass(renderPass, nextFrame);
        cmdRecorder.draw(3, 0);
        
        cmdBuffer = cmdRecorder.endRecording();
        
        tgai.execute(cmdBuffer);
        tgai.present(window, nextFrame);
    }

    return 0;
}
