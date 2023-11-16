#version 450

// Position , uv, normal and tangent same as the tga::Vertex struct
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

layout(set = 0, binding = 0) uniform CameraData{
    mat4 toWorld;
    mat4 view;
    mat4 projection;
}camera;


#define NUM_MODELS 55

// Model transform
layout(set = 1, binding = 0) uniform ModelData{
    mat4 transforms[NUM_MODELS];
} model;

layout(location = 0) out FragData{
    vec3 positionWorld;
    vec3 normal;
    vec3 tangent;
    vec2 uv;
}fragData;


void main()
{
    mat4 currentTransform = model.transforms[gl_InstanceIndex];
    // Transform vertex position from object space to world space
    vec4 worldPos = currentTransform * vec4(position, 1);
    
    // Transform normal and tangent vectors from object space to world space
    fragData.normal = mat3(currentTransform) * normal;
    fragData.tangent = mat3(currentTransform) * tangent;
    
    // Pass texture coordinates to the fragment shader
    fragData.uv = uv;
    
    // Pass transformed vertex position in world coordinates to the fragment shader
    fragData.positionWorld = worldPos.xyz;
    
    // Calculate final vertex position in clip space
    gl_Position = camera.projection * camera.view * worldPos;
}


