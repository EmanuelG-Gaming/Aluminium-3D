#version 300 es
precision mediump float;

in vec3 vColor;
in vec2 vTextureCoords;

out vec4 outColor;

uniform sampler2D uTexture;

void main() {
     outColor = vec4(1.0, 1.0, 1.0, texture(uTexture, vTextureCoords).r) * vec4(vColor.xyz, 1.0);
}