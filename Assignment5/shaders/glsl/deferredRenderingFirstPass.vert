#version 450

// Position , uv, normal and tangent same as the tga::Vertex struct
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

layout(set = 0, binding = 0) uniform CameraData{
    vec3 cameraWorldPos;
    mat4 view;
    mat4 projection;
}camera;


//#define NUM_MODELS 55

// Model transform
layout(set = 1, binding = 0) buffer ModelData{
    mat4 transform;
} model;

// layout(set = 1, binding = 0) buffer ModelData{
//     mat4 transforms[];
// } model;

layout(location = 0) out FragData{
    vec3 positionWorld;
    vec3 normal;
    vec3 tangent;
    vec2 uv;
}fragData;


void main()
{
    // Transform vertex position from object space to world space
    vec4 worldPos = model.transform * vec4(position, 1);
    
    // Transform normal and tangent vectors from object space to world space
    fragData.normal = mat3(model.transform) * normal;
    fragData.tangent = mat3(model.transform) * tangent;
    
    // Pass texture coordinates to the fragment shader
    fragData.uv = uv;
    
    // Pass transformed vertex position in world coordinates to the fragment shader
    fragData.positionWorld = worldPos.xyz;
    
    // Calculate final vertex position in clip space
    gl_Position = camera.projection * camera.view * worldPos;
}