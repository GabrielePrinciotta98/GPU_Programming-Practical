#version 450

// Data from the vertex shader (interpolated by rasterizer)
layout(location = 0) in FragData{
    vec3 positionWorld;
    vec3 normal;
    vec3 tangent;
    vec2 uv;
}fragData;

layout(set = 0, binding = 1) uniform LightData{
    vec3 position;
    vec4 color;
}light;

// Camera also bound here to get access to the inverseView / toWorld
layout(set = 0, binding = 0) uniform CameraData{
    mat4 toWorld;
    mat4 view;
    mat4 projection;
}camera;

// Texture is per model
layout(set = 1, binding = 0) uniform sampler2D colorTex;

layout(location = 0) out vec4 fragColor;

const float kambient = 0.05;
const float kdiffuse = 1.0;
const float kspecular = 1.0;
const float glossiness = 128.0;

void main()
{
    
    vec3 cameraWorldPos = camera.toWorld[3].xyz;
    vec3 V = normalize(cameraWorldPos - fragData.positionWorld);
    vec3 L = normalize(light.position);
    vec3 N = normalize(fragData.normal);
    vec3 H = normalize(V+L);
    
    //Lambert BRDF
    float lambert = max(dot(N,L), 0.0); //max is used to take only positive values of the dot product

    //Blinn-Phong BRDF
    float blinnPhong = max(dot(N,H), 0); 
    blinnPhong = pow(blinnPhong, glossiness);


    // The decompose the light data
    vec3 lightColor = light.color.rgb * light.color.w;
    vec3 objectColor = texture(colorTex,fragData.uv).rgb;

    //ambient term
    vec3 ambient = lightColor * objectColor * kambient;
    //diffuse term
    vec3 diffuse = lambert * objectColor * lightColor * kdiffuse;
    //speculat term
    vec3 specular = blinnPhong * lightColor * kspecular;

    vec3 color = ambient + diffuse + specular;

    fragColor = vec4(color,1);
}
