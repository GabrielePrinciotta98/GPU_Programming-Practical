#version 450

layout (location = 0) in FragData{
    vec2 uv;
} fragData;

layout(set = 0, binding = 0, rgba16f) uniform image2D result;

layout(location = 0) out vec4 color;


vec3 tonemap(const vec3 x){
    return clamp((x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14)),0.0,1.0);
}
void main()
{
    vec3 col = imageLoad(result,ivec2(fragData.uv*imageSize(result))).rgb;
    color = vec4(tonemap(col),1);
}