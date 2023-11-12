#version 450

layout(location = 0) in FragData {
    vec2 uv;
} fragData;

layout(set = 0, binding = 0) uniform sampler2D colorTex;

layout(location = 0) out vec4 color;

// Function to generate random values in [0,1]
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}
// Function to generate random values in [-1,1]
float randNeg(vec2 co) {
    return 2.0 * fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453) - 1.0;
}

//tonemapping
vec3 tonemapping(const vec3 x){
return pow(clamp((x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14)),0.0,1.0),vec3(0.4545454545));
}

void main() {
    vec3 sampledColor = texture(colorTex, fragData.uv).rgb;
    vec3 col;

    if (sampledColor == vec3(0)){
        vec2 curPos = gl_FragCoord.xy;

        // Create a star pattern using noise with adjusted frequency
        float starIntensity = smoothstep(0.99, 1.0, rand(curPos));

        // Bias the majority of stars towards white
        vec3 whiteStarColor = vec3(1.0, 1.0, 1.0) * (0.0000000001 * rand(curPos));

        // Introduce variations for some stars (bluish, yellowish, reddish)
        float temperature = rand(curPos);  // Temperature between 0 and 1
        vec3 coloredStarColor = mix(vec3(0.8, 0.8, 1.0), vec3(1.0, 0.8, 0.6), temperature);

        // Blend the star pattern with the background
        vec3 backgroundColor = vec3(0.0, 0.0, 0.0);  // Black background
        vec3 starColor = mix(whiteStarColor, coloredStarColor, starIntensity);

        float bloomIntensity = 1 + randNeg(curPos);
        col = mix(backgroundColor, starColor * bloomIntensity, starIntensity);
    }
    else{
        col = sampledColor;
    }
    //color = vec4(tonemapping(col), 1.0);
    color = vec4(col, 1.0);
}



