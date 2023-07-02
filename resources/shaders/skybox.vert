#version 300 es
layout (location = 0) in vec3 position;

out vec3 vTextureCoords;

uniform mat4 view;
uniform mat4 projection;
		
void main() {
     vTextureCoords = position;  
     gl_Position = vec4(position, 1.0) * mat4(mat3(view)) * projection;
}    