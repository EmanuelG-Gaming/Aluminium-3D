#version 300 es
precision mediump float;

in vec4 vPosition;
in vec3 vNormal;
in vec2 vTextureCoords;

const vec3 lightDir = vec3(-1.0, -1.3, 0.0);
const vec4 fogColor = vec4(0.5, 0.6, 0.9, 1.0); 
const float fogIntensity = 0.0005;

struct Material {
    vec3 ambientColor;
    vec3 diffuseColor;
    vec3 specularColor;
    
    float shininess;
};
uniform Material material;
uniform bool useTextures;
uniform sampler2D theTexture;
uniform vec3 viewPosition;

out vec4 outColor;


void main() {
    // Diffuse reflection
    vec3 normal = normalize(vNormal);
    vec3 lightDirection = normalize(-lightDir);
    float dotProduct = dot(lightDirection, normal);
    float intensity = clamp(dotProduct, 0.0, 1.0);
    
    // Specular reflection
    vec3 viewDirection = normalize(viewPosition - vPosition.xyz);
    vec3 reflectDirection = reflect(-lightDirection, normal);
     
    float specularIntensity = 0.0;
    if (material.specularColor != vec3(0.0, 0.0, 0.0)) specularIntensity = pow(max(dot(viewDirection, reflectDirection), 0.0), material.shininess * 128.0);
     
    vec3 ambient = material.ambientColor;
    vec3 diffuse = material.diffuseColor * intensity;
    vec3 specular = material.specularColor * specularIntensity;
    
    vec3 color;
    if (useTextures) {
        vec4 sample = texture(theTexture, vTextureCoords);
        if (sample.a <= 0.4) discard;
        
        color = (ambient + diffuse) * sample.xyz + specular;
    } else {
        color = ambient + diffuse + specular;
    }
    
    float z = (gl_FragCoord.z / gl_FragCoord.w);
    float fog = clamp(exp(-fogIntensity * z * z), 0.05, 1.0);

    outColor = mix(fogColor, vec4(color, 1.0), fog);
    //outColor = vec4(color, 1.0);
}  