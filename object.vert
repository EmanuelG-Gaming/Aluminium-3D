#version 300 es
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 textureCoords;

out vec4 vPosition;
out vec3 vNormal;
out vec2 vTextureCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
		
void main() {
     vPosition = vec4(position.xyz, 1.0) * model;
     vNormal = normal;
     vTextureCoords = textureCoords;
      
     gl_Position = vPosition * view * projection;
}    