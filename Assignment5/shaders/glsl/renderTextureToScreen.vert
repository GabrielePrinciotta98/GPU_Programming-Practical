#version 450

layout(location = 0) in vec2 position;

layout(location = 0) out FragData{
    vec2 uv;
}fragData;

vec2[4] uvs = vec2[](   
    vec2(0,0),
    vec2(1,1),
    vec2(0,1),
    vec2(1,0)
);

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    fragData.uv = uvs[gl_VertexIndex];
}



