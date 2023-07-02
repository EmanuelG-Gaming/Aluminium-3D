#version 300 es
layout (location = 0) in vec2 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec2 textureCoords;

out vec3 vColor;
out vec2 vTextureCoords;

void main() {
    vColor = color;
    vTextureCoords = textureCoords;
    
    gl_Position = vec4(position.xy, 0.0, 1.0);
}