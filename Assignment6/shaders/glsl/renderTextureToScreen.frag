#version 450

layout(location = 0) in FragData {
    vec2 uv;
} fragData;

layout(set = 0, binding = 0) uniform sampler2D colorTex;

layout(location = 0) out vec4 color;

void main() {
    color = vec4(texture(colorTex, fragData.uv).rgb, 1.0);
}



