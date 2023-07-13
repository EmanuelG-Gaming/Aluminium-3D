#version 300 es
precision mediump float;

in vec4 vPosition;
in vec3 vNormal;
in vec2 vTextureCoords;
in vec3 vColor;

const vec4 fogColor = vec4(0.5, 0.6, 0.9, 1.0); 
const float fogIntensity = 0.0005;

out vec4 outColor;

void main() {
    float z = (gl_FragCoord.z / gl_FragCoord.w);
    float fog = clamp(exp(-fogIntensity * z * z), 0.05, 1.0);

    outColor = mix(fogColor, vec4(vColor.xyz, 1.0), fog);
}  