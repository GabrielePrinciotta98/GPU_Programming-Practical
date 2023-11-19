#version 460
#extension GL_EXT_nonuniform_qualifier: enable
// Data from the vertex shader (interpolated by rasterizer)
layout(location = 0) in FragData{
    vec3 positionWorld;
    vec3 normal;
    vec3 tangent;
    vec2 uv;
}fragData;

layout(location = 5) in DrawIndex{
    flat uint drawID;
};

layout(set = 1, binding = 1) uniform sampler2D colorTextures[]; 



//G-buffer components
layout(location = 0) out vec4 positionWorldOut;
layout(location = 1) out vec4 normalOut;
layout(location = 2) out vec4 colorOut;


void main()
{
    positionWorldOut = vec4(fragData.positionWorld, 1);
    normalOut = vec4(fragData.normal, 0);
    colorOut = vec4(texture(colorTextures[drawID], fragData.uv).rgb, 1);
}
