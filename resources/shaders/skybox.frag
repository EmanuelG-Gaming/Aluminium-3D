#version 300 es
precision mediump float;

in vec3 vTextureCoords;

uniform samplerCube skybox;

out vec4 outColor;

void main() {
    outColor = texture(skybox, vTextureCoords);
}  