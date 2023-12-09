#version 450

layout (location = 0) out FragData{
    vec2 uv;
} fragData;

void main()
{
    /*
    VertexIndex[0,1,2]
    << 1 : [0,2,4] [0b0, 0b10, 0b100]
    fragData uv : [(0,0),(2,0),(0,2)]

    gl_Position [(-1,-1),(3,-1),(-1,3)]

-1,-1 ----- 1,-1------3,-1
|            |        /
|   Screen   |      /
|            |    /
-1,1--------1,1 /
|             /
|           /
|         /
-1,2    /
|     /
|   /
| /
-1,3

    */
    //https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
    fragData.uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(fragData.uv * 2.0f -1.0f, 1.0f, 1.0f);
}
