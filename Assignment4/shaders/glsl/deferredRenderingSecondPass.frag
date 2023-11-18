#version 450
#extension GL_EXT_debug_printf : enable

// Data from the vertex shader (interpolated by rasterizer)
layout(location = 0) in FragData{
    vec2 uv;
}fragData;

// Camera also bound here to get access to the inverseView / toWorld
layout(set = 0, binding = 0) uniform CameraData{
    vec3 cameraWorldPos;
    mat4 view;
    mat4 projection;
}camera;

struct Light {
    vec3 position;
    vec4 color;
};

#define NUM_LIGHTS 144

layout(set = 0, binding = 1) uniform LightData{
    Light lights[NUM_LIGHTS];
};

layout(set = 1, binding = 0) uniform sampler2D fragWorldPosTex;
layout(set = 1, binding = 1) uniform sampler2D normalsTex;
layout(set = 1, binding = 2) uniform sampler2D colorTex;

layout(location = 0) out vec4 fragColor;

const float kambient = 0.05;
const float kdiffuse = 1.0;
const float kspecular = 0.8;
const float glossiness = 128.0;

void main()
{
    vec3 positionWorld = texture(fragWorldPosTex, fragData.uv).rgb;
    vec3 normal = texture(normalsTex, fragData.uv).rgb;
    vec3 objectColor = texture(colorTex,fragData.uv).rgb;

    //vec3 cameraWorldPos = camera.toWorld[3].xyz;

    vec3 V = normalize(camera.cameraWorldPos - positionWorld);
    vec3 N = normalize(normal);
    
    vec3 color = vec3(0.0);

    vec3 testColor = camera.cameraWorldPos;
    for (int i=0; i<NUM_LIGHTS; ++i){
        vec3 L = normalize(lights[i].position);
        vec3 H = normalize(V+L);
        
        //Lambert BRDF
        float lambert = max(dot(N,L), 0.0); //max is used to take only positive values of the dot product
        
        //Blinn-Phong BRDF
        float blinnPhong = max(dot(N,H), 0.0); 
        blinnPhong = pow(blinnPhong, glossiness);
        
        vec3 lightColor = lights[i].color.rgb * lights[i].color.w;
        testColor = (lights[i].color.rgb);
        //ambient term
        vec3 ambient = lightColor * objectColor * kambient;
        //diffuse term
        vec3 diffuse = lambert * objectColor * lightColor * kdiffuse;
        //speculat term
        vec3 specular = blinnPhong * lightColor * kspecular;

        color += ambient + diffuse + specular;
    }

    fragColor = vec4(color,1);
}


