#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <vector>
#include <array>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_opengl.h>

//#include <GLES2/gl2.h>
#include <GLES3/gl3.h>

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;
SDL_Window *windows;

// Modified Cxxdroid's load texture function
static SDL_Surface *load_surface(const char *path)
{
    SDL_Surface *img = IMG_Load(path);
    if (img == NULL)
    {
        fprintf(stderr, "IMG_Load Error: %s\n", IMG_GetError());
        return NULL;
    }
    return img;
}

struct Vec2f {
    float x, y;
    Vec2f() {}
    Vec2f(float x, float y) : x(x), y(y) {}
    Vec2f set(const Vec2f &other) {
         x = other.x;
         y = other.y;
         
         return *this;
     }
     Vec2f set_zero() {
         x = y = 0.0f;
         
         return *this;
     }
     Vec2f add(const Vec2f &other) {
         x += other.x;
         y += other.y;
         
         return *this;
     }
     Vec2f subtract(const Vec2f &other) {
         x -= other.x;
         y -= other.y;
         
         return *this;
     }
     Vec2f multiply(float scalar) {
         x *= scalar;
         y *= scalar;
         
         return *this;
     }
     Vec2f divide(float scalar) {
         x /= scalar;
         y /= scalar;
         
         return *this;
     }
     Vec2f nor() {
         return multiply(1 / len());
     }
     Vec2f perpendicular(int facing) {
         int j = facing >= 0 ? 1 : -1;
         float ax = x;
         float ay = y;
         
         x = j * ay;
         y = -j * ax;
        
         return *this;
     }
     Vec2f interpolate(Vec2f &other, float progress) {
         x = x + (other.x - x) * progress;
         y = y + (other.y - y) * progress;
        
         return *this;
     }
     
     
     float len() {
         return sqrt(x * x + y * y);
     }
     float len2() {
         return x * x + y * y;
     }
     float dst(const Vec2f &other) {
         float dx = x - other.x;
         float dy = y - other.y;
         return sqrt(dx * dx + dy * dy);
     }
     float dst2(const Vec2f &other) {
         float dx = x - other.x;
         float dy = y - other.y;
         return dx * dx + dy * dy;
     }
     float dot(const Vec2f &other) {
         return x * other.x + y * other.y;
     }
     float crs(const Vec2f &other) {
         return x * other.y - y * other.x;
     }
};

struct Vec3f {
    float x, y, z;
    Vec3f() {}
    Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3f set(const Vec3f &other) {
         x = other.x;
         y = other.y;
         z = other.z;
         
         return *this;
     }
    void set_zero() {
        x = 0;
        y = 0;
        z = 0;
    }
    float dot(const Vec3f &other) {
        return x * other.x + y * other.y + z * other.z;
    }
    Vec3f crs(const Vec3f &other) { 
        float cx = y * other.z - z * other.y;
        float cy = z * other.x - x * other.z;
        float cz = x * other.y - y * other.x;
        
        Vec3f result = Vec3f(cx, cy, cz);
        
        return result;
    }  
    float len() {
        return sqrt(x*x + y*y + z*z);
    }
    float len2() {
        return x*x + y*y + z*z;
    }
    float dst(const Vec3f &other) {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        
        return sqrt(dx*dx + dy*dy + dz*dz);
    }
    float dst2(const Vec3f &other) {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        
        return dx*dx + dy*dy + dz*dz;
    }
    Vec3f nor() {
        return multiply(1 / len());
    }
    Vec3f add(const Vec3f &other) {
        x = x + other.x;
        y = y + other.y;
        z = z + other.z;
        
        return *this;
    }
    Vec3f subtract(const Vec3f &other) {
         x -= other.x;
         y -= other.y;
         z -= other.z;
         
         return *this;
    }
    Vec3f multiply(float scalar) {
        Vec3f result;
        result.x = x * scalar;
        result.y = y * scalar;
        result.z = z * scalar;
        
        return result;
    }
    Vec3f multiply(const Vec3f &other) {
        Vec3f result;
        result.x = x * other.x;
        result.y = y * other.y;
        result.z = z * other.z;
        
        return result;
    }
    // Component-wise clamping
    Vec3f clamp(const Vec3f &min, const Vec3f &max) {
        x = std::max(min.x, std::min(max.x, x));
        y = std::max(min.y, std::min(max.y, y));
        z = std::max(min.z, std::min(max.z, z));
        
        return *this;
    }
};
struct Vec2i {
    int x, y;
};

class Mat4x4 {
    public:
       int M00 = 0,  M10 = 1,  M20 = 2,  M30 = 3,
           M01 = 4,  M11 = 5,  M21 = 6,  M31 = 7,
           M02 = 8,  M12 = 9,  M22 = 10, M32 = 11,
           M03 = 12, M13 = 13, M23 = 14, M33 = 15;
           
       float values[4 * 4] = { 0 };
       
       Mat4x4() {
           this->identity();
       }
       void identity() {
           for (int i = 0; i < 4 * 4; i++) {
               values[i] = 0.0f;
           }
           values[M00] = 1.0f;
           values[M11] = 1.0f;
           values[M22] = 1.0f;
           values[M33] = 1.0f;
       }
       void set_perspective(float fovDegrees, float zNear, float zFar, float aspectRatio) {            
           float fovR = float(1.0f / tan(fovDegrees * (M_PI / 180.0f) / 2.0f));           
           float range = zFar - zNear;
           
           identity();                    
           values[M00] = fovR / aspectRatio;            
           values[M11] = fovR;            
           
           values[M22] = -(zFar + zNear) / range;            
           values[M32] = -(2 * zFar * zNear) / range;           
           values[M23] = -1.0f;            
           values[M33] = 0.0f;
       }
       void set_orthographic(float left, float right, float bottom, float top, float near, float far) {
           identity();
           
           values[M00] = 2.0f / (right - left);
           values[M11] = 2.0f / (top - bottom);
           values[M22] = -2.0f / (far - near);
           
           values[M30] = -(right + left) / (right - left);
           values[M31] = -(top + bottom) / (top - bottom);
           values[M32] = -(far + near) / (far - near);
       }
       void set_look_at(Vec3f cameraPosition, Vec3f lookingAt, Vec3f up) {
           Vec3f fwd = Vec3f(cameraPosition.x - lookingAt.x,
                             cameraPosition.y - lookingAt.y,
                             cameraPosition.z - lookingAt.z).nor();
           
           Vec3f cameraXAxis = fwd.crs(up).nor();
           
           Vec3f cameraYAxis = cameraXAxis.crs(fwd);
           
           identity();
           
           values[M00] = cameraXAxis.x;
           values[M10] = cameraXAxis.y;
           values[M20] = cameraXAxis.z;
     
           values[M01] = cameraYAxis.x;
           values[M11] = cameraYAxis.y;
           values[M21] = cameraYAxis.z;
           
           values[M02] = fwd.x;
           values[M12] = fwd.y;
           values[M22] = fwd.z;
           
           values[M30] = -cameraXAxis.dot(cameraPosition);
           values[M31] = -cameraYAxis.dot(cameraPosition);
           values[M32] = -fwd.dot(cameraPosition);
           
       }
       
       
       void set_translation(float x, float y, float z) {
           identity();
           
           values[M30] = x;
           values[M31] = y;
           values[M32] = z;
       }
       void set_translation(Vec3f to) {
           set_translation(to.x, to.y, to.z);
       }
       
       void set_scaling(float x, float y, float z) {
           identity();
           
           values[M00] = x;
           values[M11] = y;
           values[M22] = z;
       }
       void set_scaling(Vec3f to) {
           set_scaling(to.x, to.y, to.z);
       }
       void set_scaling(float scalar) {
           set_scaling(scalar, scalar, scalar);
       }
       void set_rotationX(float radians) {
           identity();
           
           values[M11] = cos(radians);
           values[M21] = -sin(radians);
           values[M12] = sin(radians);
           values[M22] = cos(radians);
       }
       void set_rotationY(float radians) {
           identity();
           
           values[M00] = cos(radians);
           values[M20] = sin(radians);
           values[M02] = -sin(radians);
           values[M22] = cos(radians);
       }
       Mat4x4 multiply(Mat4x4 &with) {
           Mat4x4 r;
           
           int row = 0;
           int column = 0;
           for (int i = 0; i < 16; i++) {
               row = (i / 4) * 4;
               column = (i % 4);
               r.values[i] = (values[row + 0] * with.values[column + 0]) +
                             (values[row + 1] * with.values[column + 4]) +
                             (values[row + 2] * with.values[column + 8]) +
                             (values[row + 3] * with.values[column + 12]);
           }
           
           return r;
       }
       // Print the column-major matrix
       std::string to_string() {
           return 
           "[ " + std::to_string(values[M00]) + "|" + std::to_string(values[M10]) + "|" + std::to_string(values[M20]) + "|" + std::to_string(values[M30]) + "|\n  " +
                 std::to_string(values[M01]) + "|" + std::to_string(values[M11]) + "|" + std::to_string(values[M21]) + "|" + std::to_string(values[M31]) + "|\n  " +
                 std::to_string(values[M02]) + "|" + std::to_string(values[M12]) + "|" + std::to_string(values[M22]) + "|" + std::to_string(values[M32]) + "|\n  " + 
                 std::to_string(values[M03]) + "|" + std::to_string(values[M13]) + "|" + std::to_string(values[M23]) + "|" + std::to_string(values[M33]) + " ]\n";
       }
};

struct Quaternion {
    float w, x, y, z;
    Quaternion() {}
    Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    
    Quaternion set(float newX, float newY, float newZ, float newW) {
         x = newX;
         y = newY;
         z = newZ;
         w = newW;
         
         return *this;
    }
    Quaternion set(const Quaternion &other) {
         x = other.x;
         y = other.y;
         z = other.z;
         w = other.w;
        
         return *this;
    }
    
    // Roll - X rotation; Pitch - Y rotation; Yaw - Z rotation
    Quaternion from_euler(float roll, float pitch, float yaw) {
         // Abbreviations
         float cr = cos(roll * 0.5f);
         float sr = sin(roll * 0.5f);
         float cp = cos(pitch * 0.5f);
         float sp = sin(pitch * 0.5f);
         float cy = cos(yaw * 0.5f);
         float sy = sin(yaw * 0.5f);
         
         Quaternion result;
         result.w = cr * cp * cy + sr * sp * sy;
         result.x = sr * cp * cy - cr * sp * sy;
         result.y = cr * sp * cy + sr * cp * sy;
         result.z = cr * cp * sy - sr * sp * cy;
         
         return result;
    }
    // In the vector: x - roll, y - pitch, z - yaw
    Vec3f to_euler() {
         Vec3f result;
         
         float coefficient = 2 * (w * y - x * z);
         float sqrt1 = sqrt(1 + coefficient); 
         float sqrt2 = sqrt(1 - coefficient);
         
         // Wikipedia:
         result.x = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
         result.y = -M_PI / 2.0f + 2 * atan2(sqrt1, sqrt2);
         result.z = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
         
         return result;
    }
    Mat4x4 to_rotation_matrix() {
         Mat4x4 result;
         
         result.values[result.M00] = w * w + x * x - y * y - z * z;
         result.values[result.M10] = 2 * (x * y - w * z);
         result.values[result.M20] = 2 * (w * y + x * z);
         
         result.values[result.M01] = 2 * (x * y + w * z);
         result.values[result.M11] = w * w - x * x + y * y - z * z;
         result.values[result.M21] = 2 * (y * z - w * x);
         
         result.values[result.M02] = 2 * (x * z - w * y);
         result.values[result.M12] = 2 * (w * x + y * z);
         result.values[result.M22] = w * w - x * x - y * y + z * z;
         
         return result;
    }
    Quaternion conjugate() {
         x = -x;
         y = -y;
         z = -z;
         
         return *this;
    }
    Vec3f transform(const Vec3f &reference) {
         // p' = q * p * conj(q)
         Quaternion temporary1 = Quaternion(reference.x, reference.y, reference.z, 0.0f), temporary2 = Quaternion(x, y, z, w);
         temporary2 = temporary2.conjugate();
         temporary2 = temporary2.multiply_left(temporary1.x, temporary1.y, temporary1.z, 0.0f).multiply_left(x, y, z, w);
         
         Vec3f result;
         
         result.x = temporary2.x;
         result.y = temporary2.y;
         result.z = temporary2.z;
         
         return result;
    } 
    
    Quaternion multiply_left(float otherX, float otherY, float otherZ, float otherW) {
         // (e + fi + gj + hk) * (a + bi + cj + dk)
         // Where a = w, b = x, c = y, d = z
         //       e = other.w, f = other.x, g = other.y, h = other.z
         
         float newX = otherW * x + otherX * w + otherY * z - otherZ * y;
         float newY = otherW * y + otherY * w + otherZ * x - otherX * z;
         float newZ = otherW * z + otherZ * w + otherX * y - otherY * x;
         float newW = otherW * w - otherX * x - otherY * y - otherZ * z;
         
         x = newX;
         y = newY;
         z = newZ;
         w = newW;
         
         return *this;
    }
};

class Camera {
    public:
        Vec3f position;
        Vec3f lookingAt;
        Vec3f lookDirection;
        float rotationX;
        float rotationY;
        bool useRotation;
        
        Camera(float fov, float zNear, float zFar, float width, float height, bool perspective, bool useRotation) {
            position.set_zero();
            rotationX = rotationY = 0.0f;
            lookingAt = Vec3f(1.0f, 0.0f, 0.0f);
            
            this->width = width;
            this->height = height;
            
            this->fov = fov;
            this->zNear = zNear;
            this->zFar = zFar;
            this->perspective = perspective;
            this->useRotation = useRotation;
        }
        
        Camera() : Camera(90.0f, 0.1f, 1000.0f, float(SCREEN_WIDTH), float(SCREEN_HEIGHT), true, false) {
        }
        
        void update() {
            if (useRotation) {
                 lookingAt.x = position.x + cos(rotationX) * cos(rotationY);
                 lookingAt.y = position.y + sin(rotationY);
                 lookingAt.z = position.z + sin(rotationX) * cos(rotationY);
            }
            
            Vec3f up = Vec3f(0.0f, 1.0f, 0.0f);
            viewMat.set_look_at(position, lookingAt, up);
            
            if (perspective) {
                projMat.set_perspective(fov, zNear, zFar, width / height);
            } else {
                projMat.set_orthographic(-width / 2, width / 2, -height / 2, height / 2, zNear, zFar);
            }
            combined = viewMat.multiply(projMat);
        }
        void resize(float width, float height) {
            this->width = width;
            this->height = height;
        }
        Mat4x4 get_projection() {
            return projMat;
        }
        Mat4x4 get_view() {
            return viewMat;
        }
    protected:
        float fov;
        float zNear, zFar;
        float width, height;
        bool perspective;
        
        Mat4x4 viewMat;
        Mat4x4 projMat;
        Mat4x4 combined;
};
class CameraControls {
    public:
        CameraControls(Camera *camera) {
            this->camera = camera;
        }
        void handle_event(SDL_Event ev, float timeTook) {
            int w = 0, h = 0;
            SDL_GetWindowSize(windows, &w, &h);
            Vec2i click = this->get_mouse_position(ev);
           
            if (ev.type == SDL_MOUSEMOTION) {       
                float sensitivity = 0.1f;
               
                if (click.y < h / 2) {
                    camera->rotationX -= ev.motion.xrel * sensitivity * timeTook;
                    camera->rotationY -= ev.motion.yrel * sensitivity * timeTook;
                }
               
                if (camera->rotationX > 2 * M_PI) camera->rotationX = 0;
                if (camera->rotationX < 0) camera->rotationX = 2 * M_PI;
               
                if (camera->rotationY > (89.0f / 180.0f * M_PI)) camera->rotationY = (89.0f / 180.0f * M_PI);
                if (camera->rotationY < -(89.0f / 180.0f * M_PI)) camera->rotationY = -(89.0f / 180.0f * M_PI);
              
                float s = 5.0f;
                Vec3f vel = Vec3f(cos(camera->rotationX) * cos(camera->rotationY),
                                  sin(camera->rotationY),
                                  sin(camera->rotationX) * cos(camera->rotationY));
                Vec3f v = vel.multiply(s * timeTook);
                
                if (click.y > h / 2 && click.x < w / 2) {
                    camera->position.add(v);
                }
                else if (click.y > h / 2 && click.x > w / 2) {
                    Vec3f back = v.multiply(-1);
                    camera->position.add(back);
                }
            }
        }
        Vec2i get_mouse_position(SDL_Event event) {
            int dx = 0, dy = 0;
            SDL_GetMouseState(&dx, &dy);
            Vec2i result = { dx, dy };
           
            return result;
        }
    private:
        Camera *camera;
};

// ----- Rendering -----
class Shader {
    public:
       const char *vertexFile, *fragmentFile;
       
       Shader(const char *vertexFile, const char *fragmentFile) {
            this->vertexFile = vertexFile;
            this->fragmentFile = fragmentFile;
            
            std::string vertContent, fragContent;
               
            std::ifstream vert;
            std::ifstream frag;
            vert.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            frag.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            try {
                vert.open(vertexFile);
                frag.open(fragmentFile);
                
                std::stringstream stringVert, stringFrag;
                stringVert << vert.rdbuf();
                stringFrag << frag.rdbuf();
                
                vert.close();
                frag.close();
                  
                vertContent = stringVert.str();
                fragContent = stringFrag.str();
             } catch (std::ifstream::failure except) {
                printf("Couldn't open the shader file: %s\n", except.what());
             }
             
             
             this->vertex = vertContent.c_str();
             this->fragment = fragContent.c_str();
             
             
             this->load(this->vertex, this->fragment);
       }
       
       void load(const char *vertSource, const char *fragSource) {
            int check;
            char log[512];
            
            
            GLuint vert = glCreateShader(GL_VERTEX_SHADER);
        	glShaderSource(vert, 1, &vertSource, NULL);
            glCompileShader(vert);
               
            glGetShaderiv(vert, GL_COMPILE_STATUS, &check); 
            if (!check) {
                 glGetShaderInfoLog(vert, 512, NULL, log);
                 printf("%s: %s\n", this->vertexFile, log);
            }
            
            
            GLuint fragm = glCreateShader(GL_FRAGMENT_SHADER);
        	glShaderSource(fragm, 1, &fragSource, NULL);
            glCompileShader(fragm);
             
            glGetShaderiv(fragm, GL_COMPILE_STATUS, &check); 
            if (!check) {
                glGetShaderInfoLog(fragm, 512, NULL, log);
                printf("%s: %s\n", this->fragmentFile, log);
            }
               
            this->program = glCreateProgram();
            glAttachShader(this->program, vert);
            glAttachShader(this->program, fragm);
            glLinkProgram(this->program);
               
            glGetProgramiv(this->program, GL_LINK_STATUS, &check);
            if (!check) {
                glGetProgramInfoLog(this->program, 512, NULL, log);
                printf("%s\n", log);
            }
            glDeleteShader(vert);
            glDeleteShader(fragm);
       }
       void use() {
           glUseProgram(this->program);
       }
       void clear() {
           glDeleteProgram(this->program);
       }
       GLint attribute_location(const char *name) {
           return glGetAttribLocation(program, name);
       }
       GLint uniform_location(const char *name) {
           return glGetUniformLocation(program, name);
       }
       GLuint get_program() {
           return program;
       }
       
       void set_uniform_int(const char *name, int value) {
           glUniform1i(this->uniform_location(name), value);
       }
       void set_uniform_bool(const char *name, bool value) {
           glUniform1i(this->uniform_location(name), (int)value);
       }
       void set_uniform_float(const char *name, float value) {
           glUniform1f(this->uniform_location(name), value);
       }
       void set_uniform_vec2f(const char *name, float x, float y) {
           glUniform2f(this->uniform_location(name), x, y);
       }
       void set_uniform_vec3f(const char *name, float x, float y, float z) {
           glUniform3f(this->uniform_location(name), x, y, z);
       }
       void set_uniform_vec4f(const char *name, float x, float y, float z, float t) {
           glUniform4f(this->uniform_location(name), x, y, z, t);
       }
       void set_uniform_mat4(const char *name, Mat4x4 input) {
           glUniformMatrix4fv(this->uniform_location(name), 1, GL_FALSE, input.values);
       }
    protected:
       const char *vertex;
       const char *fragment;
       
       GLuint program;
};

struct MeshVertex {
    Vec3f Position;
    Vec3f Normal;
    Vec2f TextureCoords;
    
    MeshVertex(Vec3f Position = Vec3f(0.0f, 0.0f, 0.0f), Vec3f Normal = Vec3f(0.0f, 0.0f, 0.0f), Vec2f TextureCoords = Vec2f(0.0f, 0.0f)) : Position(Position), Normal(Normal), TextureCoords(TextureCoords) {}
    MeshVertex(float x = 0.0f, float y = 0.0f, float z = 0.0f, float nx = 0.0f, float ny = 0.0f, float nz = 0.0f, float tx = 0.0f, float ty = 0.0f) : Position(x, y, z), Normal(nx, ny, nz), TextureCoords(tx, ty) {}
};

struct Texture {
    public:
         Texture() : repeating(true), smooth(false) {
              textureIndex = 0;
         }
         Texture(std::string fileName) : Texture() {
              setup(fileName);
         }
         void use() {
              if (textureIndex) {
                   glActiveTexture(GL_TEXTURE0);
                   glBindTexture(GL_TEXTURE_2D, textureIndex);
              }
         }
         void dispose() {
              if (textureIndex) glDeleteTextures(1, &textureIndex);
         }
    
         void setup(const std::string &fileName) {
              
              texture = load_surface(fileName.c_str());
              if (texture == NULL) {
                   printf("Texture couldn't load.");
                   return;
              }
              
              glGenTextures(1, &textureIndex);
              glBindTexture(GL_TEXTURE_2D, textureIndex);
              
              glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture->w, texture->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture->pixels);
              glGenerateMipmap(GL_TEXTURE_2D);
                   
              if (repeating) {
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
              } else {
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
              }
              
              if (smooth) {
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              } else {
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
              }
              
              glBindTexture(GL_TEXTURE_2D, 0);
              SDL_FreeSurface(texture);
              
         }
    protected:
         // The texture data
         SDL_Surface *texture;
         
         GLuint textureIndex;
         
         // Texture parameters
         bool repeating;
         bool smooth;
};

struct Material {
    Vec3f ambientColor, diffuseColor, specularColor;
    float shininess;
    
    Material(Vec3f ambient = Vec3f(1.0f, 1.0f, 1.0f), Vec3f diffuse = Vec3f(1.0f, 1.0f, 1.0f), Vec3f specular = Vec3f(0.0f, 0.0f, 0.0f), float shininess = 0.1f) : ambientColor(ambient), diffuseColor(diffuse), specularColor(specular), shininess(shininess) {}
};
struct MeshStructure {
    std::vector<MeshVertex> vertices;
    std::vector<GLuint> indices;
    
    MeshStructure() {}
    MeshStructure(std::vector<MeshVertex> vertices, std::vector<GLuint> indices) : vertices(vertices), indices(indices) {}
};

struct Mesh {
    public:
         Mesh() : useTextures(false) {
              texture = new Texture();
              vbo = ibo = vao = 0;
         }
         void set_texture(const std::string &fileName) {
              texture->setup(fileName);
              useTextures = true;
         }
         void unuse_texture() { useTextures = false; }
         void unuse_indices() { useIndices = false; }
         
         void set_material(const Material &newMaterial) {
              material = newMaterial;
         }
         
         MeshStructure get_structure() { return this->structure; }
         void set_structure(const MeshStructure &newStructure) {
              structure.vertices.clear();
              structure.indices.clear();
              
              for (auto &vertex : newStructure.vertices) {
                   structure.vertices.push_back(vertex);
              }
              if (useIndices) {
                   for (auto &index : newStructure.indices) {
                        structure.indices.push_back(index);
                   }
              }
              
              post_setup();
         }
         void translate(float x, float y, float z) {
              translation.set_translation(x, y, z);
         }
         void translate(const Vec3f &position) {
              translation.set_translation(position);
         }
         void scale(float x, float y, float z) {
              scaling.set_scaling(x, y, z);
         }
         void scale(const Vec3f &scl) {
              scaling.set_scaling(scl);
         }
         void rotate(Quaternion quat) {
              rotation = quat.to_rotation_matrix();
         }
         
         void render(Shader *shader) {
              if (!vao) return;
              
              if (useTextures) texture->use();
              
              Mat4x4 model;
              model = model.multiply(translation);
              model = model.multiply(rotation);
              model = model.multiply(scaling);
              shader->set_uniform_mat4("model", model);
              
              shader->set_uniform_vec3f("material.ambientColor", material.ambientColor.x, material.ambientColor.y, material.ambientColor.z);
              shader->set_uniform_vec3f("material.diffuseColor", material.diffuseColor.x, material.diffuseColor.y, material.diffuseColor.z);
              shader->set_uniform_vec3f("material.specularColor", material.specularColor.x, material.specularColor.y, material.specularColor.z);
              shader->set_uniform_float("material.shininess", material.shininess);
              shader->set_uniform_bool("useTextures", useTextures);
              
              glBindVertexArray(vao);
              if (useIndices) glDrawElements(GL_TRIANGLES, structure.indices.size(), GL_UNSIGNED_INT, 0);
              else glDrawArrays(GL_TRIANGLES, 0, structure.vertices.size());
           
              glBindVertexArray(0);
              
         }
         
         void dispose() {
              if (vbo) glDeleteBuffers(1, &vbo);
              if (ibo && useIndices) glDeleteBuffers(1, &ibo);
              if (vao) glDeleteVertexArrays(1, &vao);
              if (useTextures) texture->dispose();
         }
    private:
        void post_setup() {
              if (structure.vertices.empty() || (structure.indices.empty() && useIndices)) return;
              
              if (!vao) glGenVertexArrays(1, &vao);
              glBindVertexArray(vao);
           
              if (!vbo) glGenBuffers(1, &vbo);
              if (!ibo && useIndices) glGenBuffers(1, &ibo);
              glBindBuffer(GL_ARRAY_BUFFER, vbo);
              glBufferData(GL_ARRAY_BUFFER, structure.vertices.size() * sizeof(MeshVertex), structure.vertices.data(), GL_STATIC_DRAW); 
              
              if (useIndices) {
                  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
                  glBufferData(GL_ELEMENT_ARRAY_BUFFER, structure.indices.size() * sizeof(GLuint), structure.indices.data(), GL_STATIC_DRAW);         
              }
              
              // Position
              GLint position = 0;
              glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*) offsetof(MeshVertex, Position));
              glEnableVertexAttribArray(position);
              
              // Normal
              GLint normal = 1;
              glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*) offsetof(MeshVertex, Normal));
              glEnableVertexAttribArray(normal);
              
              // Texture coordinates
              GLint textureCoords = 2;
              glVertexAttribPointer(textureCoords, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*) offsetof(MeshVertex, TextureCoords));
              glEnableVertexAttribArray(textureCoords);
           
              
              glBindVertexArray(0);
              glBindBuffer(GL_ARRAY_BUFFER, 0);
              glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
              
              glDisableVertexAttribArray(position);
              glDisableVertexAttribArray(normal);
              glDisableVertexAttribArray(textureCoords);
              
       }
        
    protected:
         MeshStructure structure;
         Texture *texture;
         bool useTextures;
         bool useIndices = true;
         
         Mat4x4 scaling, rotation, translation;
         
         Material material;
         GLuint vbo, ibo;
         GLuint vao;
};

namespace Materials {
    Material aluminium, gold, copper,
             grass;
    
    void load() {
         aluminium = Material(Vec3f(0.39f, 0.39f, 0.39f), Vec3f(0.504f, 0.504f, 0.504f), Vec3f(0.508f, 0.508f, 0.508f), 0.2f);
         gold = Material(Vec3f(0.39f, 0.34f, 0.22f), Vec3f(0.75f, 0.6f, 0.22f), Vec3f(0.62f, 0.55f, 0.36f), 0.2f);
         copper = Material(Vec3f(0.39f, 0.30f, 0.22f), Vec3f(0.73f, 0.30f, 0.11f), Vec3f(0.45f, 0.33f, 0.28f), 0.2f);
         
         grass = Material(Vec3f(0.7f, 0.7f, 0.7f), Vec3f(1.0f, 1.0f, 1.0f), Vec3f(0.0f, 0.0f, 0.0f));
    }
};

namespace MeshGenerator {
    MeshStructure get_sphere_mesh(int latitudeLines, int longitudeLines) {
         std::vector<MeshVertex> vertices;
         std::vector<GLuint> indices;
          
         float latitudeSpacing = M_PI / latitudeLines;
         float longitudeSpacing = 2 * M_PI / longitudeLines;
         
         // Vertices
         for (int i = 0; i <= latitudeLines; i++) {
              for (int j = 0; j <= longitudeLines; j++) {
                   // Horizontal rotation
                   float theta = j * longitudeSpacing;
                   // Vertical rotation
                   float phi = M_PI / 2.0f - i * latitudeSpacing;
                    
                   Vec3f position = Vec3f(0.0f, 0.0f, 0.0f), normal = Vec3f(0.0f, 0.0f, 0.0f);
                   Vec2f textureCoords = Vec2f(0.0f, 0.0f);
                    
                   position.x = cos(phi) * cos(theta);
                   position.y = cos(phi) * sin(theta);
                   position.z = sin(phi);
                    
                   normal = Vec3f(position);
                   textureCoords.x = 0.5f + atan2(position.z, position.x) / (2 * M_PI);
                   textureCoords.y = 0.5f + asin(position.y) / M_PI;
                   
                    
                   MeshVertex vertex = MeshVertex(position, normal, textureCoords);
                   vertices.push_back(vertex);
              }
         }
         
         // Indices
         int k1, k2;
         for (int i = 0; i < latitudeLines; i++) {
              k1 = i * (latitudeLines + 1);
              k2 = k1 + latitudeLines + 1;
              for (int j = 0; j < longitudeLines; j++, k1++, k2++) {
                   if (i != 0) {
                        indices.push_back(k1);
                        indices.push_back(k2);
                        indices.push_back(k1 + 1);
                   }
                   if (i != (latitudeLines - 1)) {
                        indices.push_back(k1 + 1);
                        indices.push_back(k2); 
                        indices.push_back(k2 + 1);
                   }
              }
         }
         
         return MeshStructure(vertices, indices);
     }
     
     MeshStructure get_cuboid_mesh(float width, float height, float depth) {
         // Vertices 
         std::vector<MeshVertex> vertices {
              // Position, normal, texture coordinates
              // Front
              MeshVertex(-0.5, -0.5, 0.5,  0.0, 0.0, 1.0,  0.0, 0.0),
              MeshVertex(0.5, -0.5, 0.5,  0.0, 0.0, 1.0,  width, 0.0),
              MeshVertex(0.5, 0.5, 0.5,  0.0, 0.0, 1.0,  width, height),
              MeshVertex(-0.5, 0.5, 0.5,  0.0, 0.0, 1.0,  0.0, height),
              // Top
              MeshVertex(-0.5, 0.5, 0.5,  0.0, 1.0, 0.0,  0.0, 0.0),
              MeshVertex(0.5, 0.5, 0.5,  0.0, 1.0, 0.0,  width, 0.0),
              MeshVertex(0.5, 0.5, -0.5,  0.0, 1.0, 0.0,  width, depth),
              MeshVertex(-0.5, 0.5, -0.5,  0.0, 1.0, 0.0,  0.0, depth),
              // Back
              MeshVertex(0.5, -0.5, -0.5,  0.0, 0.0, -1.0,  0.0, 0.0),
              MeshVertex(-0.5, -0.5, -0.5,  0.0, 0.0, -1.0,  width, 0.0),
              MeshVertex(-0.5, 0.5, -0.5,  0.0, 0.0, -1.0,  width, height),
              MeshVertex(0.5, 0.5, -0.5,  0.0, 0.0, -1.0,  0.0, height),
              // Bottom
              MeshVertex(-0.5, -0.5, -0.5,  0.0, -1.0, 0.0,  0.0, 0.0),
              MeshVertex(0.5, -0.5, -0.5,  0.0, -1.0, 0.0,  width, 0.0),
              MeshVertex(0.5, -0.5, 0.5,  0.0, -1.0, 0.0,  width, depth),
              MeshVertex(-0.5, -0.5, 0.5, 0.0, -1.0, 0.0,  0.0, depth),
              // Left
              MeshVertex(-0.5, -0.5, -0.5,  -1.0, 0.0, 0.0,  0.0, 0.0),
              MeshVertex(-0.5, -0.5, 0.5,  -1.0, 0.0, 0.0,  depth, 0.0),
              MeshVertex(-0.5, 0.5, 0.5,  -1.0, 0.0, 0.0,  depth, height),
              MeshVertex(-0.5, 0.5, -0.5,  -1.0, 0.0, 0.0,  0.0, height),
              // Right
              MeshVertex(0.5, -0.5, 0.5,  1.0, 0.0, 0.0,  0.0, 0.0),
              MeshVertex(0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  depth, 0.0),
              MeshVertex(0.5, 0.5, -0.5,  1.0, 0.0, 0.0, depth, height),  
              MeshVertex(0.5, 0.5, 0.5,  1.0, 0.0, 0.0,  0.0, height)
         };
         for (auto &vertex : vertices) {
              vertex.Position.x *= width;
              vertex.Position.y *= height;
              vertex.Position.z *= depth;
              vertex.Position.x *= 2;
              vertex.Position.y *= 2;
              vertex.Position.z *= 2;
              
              vertex.TextureCoords.x *= 2;
              vertex.TextureCoords.y *= 2;
              
         }
         // Indices
         std::vector<GLuint> indices = {
              // Front
              0, 1, 2,
              2, 3, 0,
              // Top
              4, 5, 6,
              6, 7, 4,
              // Back
              8, 9, 10,
              10, 11, 8,
              // Bottom
              12, 13, 14,
              14, 15, 12,
              // Left
              16, 17, 18,
              18, 19, 16,
              // Right
              20, 21, 22,
              22, 23, 20
         };
         
         return MeshStructure(vertices, indices);
     }
     
     bool compare(std::string first, std::string other) {
         return !first.compare(other);
     }
     bool is_line_empty(std::string line) {
         bool empty = compare(line, "") ||
                      compare(line, " ");
                          
         return empty;
     }
     
     void load_from_files(Mesh *source, const std::string &objName, const std::string &mtlName) {
         std::ifstream objRead(objName);
         std::vector<Vec3f> positions, normals;
         std::vector<GLuint> vertexIndices, uvIndices, normalIndices;
         
         if (!objRead.is_open() || objRead.fail()) {
              printf("Couldn't open the .obj file.");
              return;
         }   
       
         std::string line; 
         while (std::getline(objRead, line)) {
              if (is_line_empty(line)) {
                  continue;
              }
              
              std::istringstream stream(line);
              std::string key;
              stream >> key;
              
              // Vertex position
              if (compare(key, "v")) {
                  Vec3f position;
                  stream >> position.x >> position.y >> position.z;
                  
                  positions.push_back(position);
              }
              // Vertex normal
              else if (compare(key, "vn")) {
                  Vec3f normal;
                  stream >> normal.x >> normal.y >> normal.z;
                  
                  normals.push_back(normal);
              }
              // Face indices
              else if (compare(key, "f")) {
                  std::string faceIndices[3];
                  stream >> faceIndices[0] >> faceIndices[1] >> faceIndices[2];
                  
                  for (int i = 0; i < 3; i++) {
                       GLuint indexPosition, indexUV, indexNormal;
                       sscanf(faceIndices[i].c_str(), "%d/%d/%d", &indexPosition, &indexUV, &indexNormal);
                       indexPosition--; indexUV--; indexNormal--;
                       
                       vertexIndices.push_back(indexPosition);
                       uvIndices.push_back(indexUV);
                       normalIndices.push_back(indexNormal);
                  }
              }
          }
          
          // Build model
          MeshStructure structure;
          for (int i = 0; i < vertexIndices.size(); i++) {
              GLuint indexPosition = vertexIndices.at(i);
              GLuint indexUV = uvIndices.at(i);
              GLuint indexNormal = normalIndices.at(i);
              
              MeshVertex vertex = MeshVertex(positions.at(indexPosition), normals.at(indexNormal));
              structure.vertices.push_back(vertex);
          }
          source->unuse_indices();
          source->set_structure(structure);
     } 
     
     
     // Loads meshes from a single model, based on the "g" qualifier
     std::vector<MeshStructure> load_meshes_from_file(const std::string &objName) {
         std::vector<MeshStructure> result;
         std::ifstream objRead(objName);
         
         std::vector<Vec3f> positions, normals;
         std::vector<GLuint> vertexIndices, uvIndices, normalIndices;
         
         if (!objRead.is_open() || objRead.fail()) {
              printf("Couldn't open the .obj file.");
              return result;
         }   
       
         std::string line; 
         while (std::getline(objRead, line)) {
              if (is_line_empty(line)) {
                  continue;
              }
              
              std::istringstream stream(line);
              std::string key;
              stream >> key;
              
              // Vertex position
              if (compare(key, "v")) {
                  Vec3f position;
                  stream >> position.x >> position.y >> position.z;
                  
                  positions.push_back(position);
              }
              // Vertex normal
              else if (compare(key, "vn")) {
                  Vec3f normal;
                  stream >> normal.x >> normal.y >> normal.z;
                  
                  normals.push_back(normal);
              }
              // Face indices
              else if (compare(key, "f")) {
                  std::string faceIndices[3];
                  stream >> faceIndices[0] >> faceIndices[1] >> faceIndices[2];
                  
                  for (int i = 0; i < 3; i++) {
                       GLuint indexPosition, indexUV, indexNormal;
                       sscanf(faceIndices[i].c_str(), "%d/%d/%d", &indexPosition, &indexUV, &indexNormal);
                       indexPosition--; indexUV--; indexNormal--;
                       
                       vertexIndices.push_back(indexPosition);
                       uvIndices.push_back(indexUV);
                       normalIndices.push_back(indexNormal);
                  }
              }
              
              // Build mesh
              if (compare(key, "g")) {
                  MeshStructure structure;
                  for (int i = 0; i < vertexIndices.size(); i++) {
                       GLuint indexPosition = vertexIndices.at(i);
                       GLuint indexUV = uvIndices.at(i);
                       GLuint indexNormal = normalIndices.at(i);
              
                       MeshVertex vertex = MeshVertex(positions.at(indexPosition), normals.at(indexNormal));
                       structure.vertices.push_back(vertex);
                  }
                  if (!vertexIndices.empty()) {
                       result.push_back(structure);
                  }
                  vertexIndices.clear();
                  uvIndices.clear();
                  normalIndices.clear();
              }
         }
         
         // Build the final mesh
         MeshStructure structure;
         for (int i = 0; i < vertexIndices.size(); i++) {
              GLuint indexPosition = vertexIndices.at(i);
              GLuint indexUV = uvIndices.at(i);
              GLuint indexNormal = normalIndices.at(i);
              
              MeshVertex vertex = MeshVertex(positions.at(indexPosition), normals.at(indexNormal));
              structure.vertices.push_back(vertex);
         }
         if (!vertexIndices.empty()) {
              result.push_back(structure);
         } 
            
         return result;
     }
     
     
     Vec3f generate_bounding_box_sizes(const MeshStructure &source) {
          MeshStructure structure = source;
          if (structure.vertices.empty()) return Vec3f(0.0f, 0.0f, 0.0f);
          
          Vec3f minimum, maximum;
          minimum = maximum = Vec3f(structure.vertices.at(0).Position);
            
          for (auto &vertex : structure.vertices) {
              // X coordinates
              if (vertex.Position.x < minimum.x) minimum.x = vertex.Position.x;
              if (vertex.Position.x > maximum.x) maximum.x = vertex.Position.x;
              
              // Y coordinates
              if (vertex.Position.y < minimum.y) minimum.y = vertex.Position.y;
              if (vertex.Position.y > maximum.y) maximum.y = vertex.Position.y;
                  
              // Z coordinates
              if (vertex.Position.z < minimum.z) minimum.z = vertex.Position.z;
              if (vertex.Position.z > maximum.z) maximum.z = vertex.Position.z;
          }
          
          float width = (maximum.x - minimum.x) / 2.0f;
          float height = (maximum.y - minimum.y) / 2.0f;
          float depth = (maximum.z - minimum.z) / 2.0f;
          
          return Vec3f(width, height, depth);
     }
     Vec3f generate_bounding_box_centroid(const MeshStructure &source) {
          MeshStructure structure = source;
          if (structure.vertices.empty()) return Vec3f(0.0f, 0.0f, 0.0f);
          
          Vec3f minimum, maximum;
          minimum = maximum = Vec3f(structure.vertices.at(0).Position);
            
          for (auto &vertex : structure.vertices) {
              // X coordinates
              if (vertex.Position.x < minimum.x) minimum.x = vertex.Position.x;
              if (vertex.Position.x > maximum.x) maximum.x = vertex.Position.x;
              
              // Y coordinates
              if (vertex.Position.y < minimum.y) minimum.y = vertex.Position.y;
              if (vertex.Position.y > maximum.y) maximum.y = vertex.Position.y;
                  
              // Z coordinates
              if (vertex.Position.z < minimum.z) minimum.z = vertex.Position.z;
              if (vertex.Position.z > maximum.z) maximum.z = vertex.Position.z;
          }
          
          float x = (maximum.x + minimum.x) / 2.0f;
          float y = (maximum.y + minimum.y) / 2.0f;
          float z = (maximum.z + minimum.z) / 2.0f;
          
          return Vec3f(x, y, z);
     }
};

// ----- Physics -----
static struct Collider;
static struct SphereCollider;
static struct BoxCollider;

static struct Manifold;

struct ManifoldPoints {
      // Furthest point from object1 to object2
      Vec3f AtoB;
      // Furthest point from object2 to object1
      Vec3f BtoA;
      
      // BtoA - AtoB, normalized
      Vec3f normal;
      // Length of BtoA - AtoB, not normalized
      float depth;
      // Wheter or not the objects have collided
      bool collided = false;
      
      ManifoldPoints() {}
      // Use if the objects have collided
      ManifoldPoints(Vec3f AtoB, Vec3f BtoA, Vec3f normal, float depth) : AtoB(AtoB), BtoA(BtoA), normal(normal), depth(depth), collided(true) {}
};


struct Transform {
      Vec3f position;
      Quaternion rotation;
      
      Transform() { 
           this->reset();
      }
      void reset() {
           position = Vec3f(0.0f, 0.0f, 0.0f);
           rotation = rotation.from_euler(0.0f, 0.0f, 0.0f);
      }
};

namespace CollisionDetection {
     ManifoldPoints sphere_sphere(SphereCollider *sphere, Transform *transform, SphereCollider *other, Transform *otherTransform);
     ManifoldPoints sphere_AABB(SphereCollider *sphere, Transform *transform, BoxCollider *other, Transform *otherTransform);
     ManifoldPoints sphere_OBB(SphereCollider *sphere, Transform *transform, BoxCollider *other, Transform *otherTransform);
};

struct Collider {
     // Collider-collider
     virtual ManifoldPoints test(Transform *transform, Collider *collider, Transform *otherTransform) = 0;
     // Collider-sphere
     virtual ManifoldPoints test(Transform *transform, SphereCollider *sphereCollider, Transform *otherTransform) = 0;
     // Collider-AABB
     virtual ManifoldPoints test(Transform *transform, BoxCollider *aabbCollider, Transform *otherTransform)  = 0;
};

struct SphereCollider : Collider {
     float radius;
     SphereCollider() {}
     SphereCollider(float radius) : radius(radius) {}
     
     // Sphere-collider
     ManifoldPoints test(Transform *transform, Collider *collider, Transform *otherTransform) override { 
           return collider->test(otherTransform, this, transform); 
     }
     // Sphere-sphere
     ManifoldPoints test(Transform *transform, SphereCollider *sphereCollider, Transform *otherTransform) override { 
           return CollisionDetection::sphere_sphere(this, transform, sphereCollider, otherTransform);
     }
     // Sphere-AABB
     ManifoldPoints test(Transform *transform, BoxCollider *aabbCollider, Transform *otherTransform) override {
           return CollisionDetection::sphere_OBB(this, transform, aabbCollider, otherTransform);
     }
};

struct BoxCollider : Collider {
     float width, height, depth;
     BoxCollider() {}
     BoxCollider(float width, float height, float depth) : width(width), height(height), depth(depth) {}
     BoxCollider(Vec3f size) : width(size.x), height(size.y), depth(size.z) {}
     
     // AABB-collider
     ManifoldPoints test(Transform *transform, Collider *collider, Transform *otherTransform) override { 
           return collider->test(otherTransform, this, transform); 
     }
     // AABB-sphere
     ManifoldPoints test(Transform *transform, SphereCollider *sphereCollider, Transform *otherTransform) override { 
           // Reuse sphere code
           ManifoldPoints points = sphereCollider->test(otherTransform, this, transform);
           
           // Swap the points, so that the collision might not break
           Vec3f temporary = points.AtoB;
           points.AtoB = points.BtoA;
           points.BtoA = temporary;
           points.normal = points.normal.multiply(-1);
           
           return points;
     }
     // No AABB-AABB
     ManifoldPoints test(Transform *transform, BoxCollider *aabbCollider, Transform *otherTransform) override { 
           return ManifoldPoints();
     }
     
     bool contains_point(Transform *transform, const Vec3f &point) {
           return (point.x >= transform->position.x - width && point.x <= transform->position.x + width) &&
                  (point.y >= transform->position.y - height && point.y <= transform->position.y + height) &&
                  (point.z >= transform->position.z - depth && point.z <= transform->position.z + depth);
     }
};


namespace CollisionDetection {
     ManifoldPoints sphere_sphere(SphereCollider *sphere, Transform *transform, SphereCollider *other, Transform *otherTransform) {
          Vec3f position = transform->position;
          Vec3f otherPosition = otherTransform->position;
          
          float distance = position.dst2(otherPosition);
          float radius = sphere->radius;
          float radiusOther = other->radius;
          bool intersecting = distance <= (radius + radiusOther) * (radius + radiusOther);
          if (!intersecting || distance < 0.00001f) {
              return ManifoldPoints();
          }
          
          // Points on the spheres
          Vec3f AtoB = Vec3f(otherPosition).subtract(position).nor();
          Vec3f BtoA = Vec3f(AtoB).multiply(-1);
          AtoB = AtoB.multiply(radiusOther);
          BtoA = BtoA.multiply(radius);
          AtoB = AtoB.add(position);
          BtoA = BtoA.add(otherPosition);
          
         
          Vec3f direction = Vec3f(BtoA).subtract(AtoB);
          Vec3f normal = Vec3f(direction).nor();
          return ManifoldPoints(AtoB, BtoA, normal, direction.len());
     }
     ManifoldPoints sphere_AABB(SphereCollider *sphere, Transform *transform, BoxCollider *other, Transform *otherTransform) {
          // Sphere and AABB
          Vec3f position = transform->position;
          Vec3f otherPosition = otherTransform->position;
           
          Vec3f lower = Vec3f(-other->width, -other->height, -other->depth);
          Vec3f upper = Vec3f(other->width, other->height, other->depth);
          
          Vec3f closest = Vec3f(position).subtract(otherPosition).clamp(lower, upper).add(otherPosition);
          float distance2 = closest.dst2(position);
          bool intersecting = distance2 <= (sphere->radius * sphere->radius);
          if (!intersecting || distance2 < 0.00001f) return ManifoldPoints();
          
          Vec3f AtoB = Vec3f(closest).subtract(position).nor().multiply(sphere->radius).add(position);
          Vec3f BtoA = closest;
                   
          Vec3f dir = Vec3f(BtoA).subtract(AtoB);
          Vec3f normal = Vec3f(dir).nor();
          return ManifoldPoints(AtoB, BtoA, normal, dir.len());
     }
     ManifoldPoints sphere_OBB(SphereCollider *sphere, Transform *transform, BoxCollider *other, Transform *otherTransform) {
          // Sphere and oriented bounding box
          Vec3f position = transform->position;
          Vec3f otherPosition = otherTransform->position;
          Quaternion rotation = otherTransform->rotation;
          Quaternion inverse = Quaternion(rotation).conjugate();
          
          Vec3f lower = Vec3f(-other->width, -other->height, -other->depth);
          Vec3f upper = Vec3f(other->width, other->height, other->depth);
          
          Vec3f centerUnrotated = Vec3f(position).subtract(otherPosition);
          centerUnrotated = inverse.transform(centerUnrotated);
          Vec3f closestUnrotated = Vec3f(centerUnrotated).clamp(lower, upper).add(otherPosition);
          centerUnrotated = centerUnrotated.add(otherPosition);
          
          float distance2 = closestUnrotated.dst2(centerUnrotated);
          bool intersecting = distance2 <= (sphere->radius * sphere->radius);
          if (!intersecting || distance2 < 0.00001f) return ManifoldPoints();
          
          Vec3f closestRotated = Vec3f(closestUnrotated).subtract(otherPosition);
          closestRotated = rotation.transform(closestRotated);
          closestRotated = closestRotated.add(otherPosition);
          
          
          Vec3f AtoB = Vec3f(closestRotated).subtract(position).nor().multiply(sphere->radius).add(position);
          Vec3f BtoA = closestRotated;
                   
          Vec3f dir = Vec3f(BtoA).subtract(AtoB);
          Vec3f normal = Vec3f(dir).nor();
          return ManifoldPoints(AtoB, BtoA, normal, dir.len());
     }
};

struct Object {
    Transform *transform;
    Collider *collider = 0;
    int index = 0;
    bool immovable = false;
    Mesh *mesh;
    Manifold *colliding = 0;
    Vec3f collidingNormal = Vec3f(0.0f, 0.0f, 0.0f);
    
    Object() {
        transform = new Transform();
        mesh = new Mesh();
    }
    Object(bool immovable) : immovable(immovable) {
        transform = new Transform();
        mesh = new Mesh();
    }
    
    
    void place(const Vec3f &to) {
        transform->position = to;
    }
    
    void place(float x, float y, float z) {
        transform->position = Vec3f(x, y, z);
    }
    void place_offset(float x, float y, float z) {
        transform->position.add(Vec3f(x, y, z));
    }
    
    void rotate(float roll, float pitch, float yaw) {
        transform->rotation = transform->rotation.from_euler(roll, pitch, yaw);
    }
    void set_immovable(bool to) {
        immovable = to;
    }
    
    
    void set_material(const Material &material) {
        mesh->set_material(material);
    }
    void set_texture(const std::string &fileName) {
        mesh->set_texture(fileName);
    }
    void set_structure(const MeshStructure &structure) {
        mesh->set_structure(structure);
    }
    
    bool has_collided() { return (colliding); }
};

struct RigidBody : public Object {
    Vec3f velocity;
    Vec3f force;
    float mass = 1.0f;
    
    RigidBody() : Object() {
          velocity = Vec3f(0.0f, 0.0f, 0.0f);
          force = Vec3f(0.0f, 0.0f, 0.0f);
    }
    RigidBody(bool immovable) : RigidBody() {
         this->immovable = immovable;
    }
    
    void apply_velocity(float x, float y, float z) {
         velocity.x += x;
         velocity.y += y;
         velocity.z += z;
    }
    void apply_force(float x, float y, float z) {
         force.x += x;
         force.y += y;
         force.z += z;
    }
    void set_mass(float newMass) {
         mass = newMass;
    }
};
struct SphereObject : public RigidBody {
    float radius = 0.1f;
    SphereObject() : RigidBody() {
         this->collider = new SphereCollider(radius);
         
         this->set_structure(MeshGenerator::get_sphere_mesh(15, 15));
         this->mesh->scale(radius, radius, radius); 
    }
    SphereObject(float radius) : RigidBody() {
         this->radius = radius;
         this->collider = new SphereCollider(radius);
         
         
         this->set_structure(MeshGenerator::get_sphere_mesh(15, 15));
         this->mesh->scale(this->radius, this->radius, this->radius); 
    }
};
struct BoxObject : public RigidBody {
    float width = 1.0f, height = 1.0f, depth = 1.0f;
    
    BoxObject() : RigidBody() {
         this->collider = new BoxCollider(width, height, depth);
         
         this->set_structure(MeshGenerator::get_cuboid_mesh(width, height, depth));
    }
    BoxObject(float width, float height, float depth) : RigidBody() {
         this->width = width;
         this->height = height;
         this->depth = depth;
         this->collider = new BoxCollider(width, height, depth);
         
         this->set_structure(MeshGenerator::get_cuboid_mesh(width, height, depth));
    }
};
struct ModelObject : public RigidBody {
    ModelObject() : RigidBody() {}
    ModelObject(const std::string &objFile) : RigidBody() {
         MeshGenerator::load_from_files(this->mesh, objFile, ""); // No material file
         this->collider = new BoxCollider(MeshGenerator::generate_bounding_box_sizes(this->mesh->get_structure()));
    }
    ModelObject(const MeshStructure &structure) : RigidBody() {
         Vec3f centroid = MeshGenerator::generate_bounding_box_centroid(structure);
         MeshStructure newStructure;
         for (auto &vertex : structure.vertices) {
              newStructure.vertices.push_back(MeshVertex(Vec3f(vertex.Position).subtract(centroid), vertex.Normal));
         }
         
         this->mesh->unuse_indices();
         this->mesh->set_structure(newStructure);
         this->collider = new BoxCollider(MeshGenerator::generate_bounding_box_sizes(structure));
         this->place(centroid);
    }
};

struct Manifold {
     Object *object1;
     Object *object2;
     ManifoldPoints points;
     Manifold() {}
     Manifold(Object *object1, Object *object2, ManifoldPoints points) : object1(object1), object2(object2), points(points) {}
};
struct Solver {
     virtual void solve(std::vector<Manifold> &manifolds) {};
};
struct PositionSolver : Solver {
     void solve(std::vector<Manifold> &manifolds) override {
           for (auto &manifold : manifolds) {
                // Avoid updating if the objects have a very small collisison depth
                if (manifold.points.depth < 0.00001f) continue;
                
                Object *object1 = manifold.object1;
                Object *object2 = manifold.object2;
                
                int immovable1 = (int)object1->immovable;
                int immovable2 = (int)object2->immovable;
                
                Vec3f normal = manifold.points.normal; //Vec3f(manifold.points.BtoA).subtract(manifold.points.AtoB);
                Vec3f displacement = Vec3f(normal).multiply(1 / (float)std::max(1, immovable1 + immovable2));
                displacement = displacement.multiply(manifold.points.depth);
                displacement = displacement.multiply(0.5f);
                
                object1->transform->position.x -= displacement.x * (1 - immovable1);
                object1->transform->position.y -= displacement.y * (1 - immovable1);
                object1->transform->position.z -= displacement.z * (1 - immovable1);
                
                object2->transform->position.x += displacement.x * (1 - immovable2);
                object2->transform->position.y += displacement.y * (1 - immovable2);
                object2->transform->position.z += displacement.z * (1 - immovable2);
           }
     }
};
struct ElasticImpulseSolver : Solver {
     void solve(std::vector<Manifold> &manifolds) override {
           for (auto &manifold : manifolds) {
               // Avoid updating if the objects have a very small collisison depth 
               if (manifold.points.depth < 0.00001f) continue;
                
               RigidBody *object1 = (RigidBody*) manifold.object1;
               RigidBody *object2 = (RigidBody*) manifold.object2;
               
               Vec3f position1 = object1->transform->position;
               Vec3f position2 = object2->transform->position;
               
               Vec3f normal = manifold.points.normal; //Vec3f(position2).subtract(position1).nor();
               Vec3f gradientVelocity = Vec3f(object1->velocity).subtract(object2->velocity);
                          
               float dot = normal.dot(gradientVelocity);
               float j = 2 * dot / (object1->mass + object2->mass);
                
                
               if (!object1->immovable) {   
                    object1->velocity.x -= j * normal.x * object2->mass;
                    object1->velocity.y -= j * normal.y * object2->mass;
                    object1->velocity.z -= j * normal.z * object2->mass;
               }
               
               if (!object2->immovable) {                   
                    object2->velocity.x += j * normal.x * object1->mass;
                    object2->velocity.y += j * normal.y * object1->mass;
                    object2->velocity.z += j * normal.z * object1->mass;
               }
          }
     }
};

class PhysicsLevel {
    std::vector<Object*> objects;
    std::vector<Solver*> solvers;
    std::vector<Manifold> manifolds;
             
    Vec3f gravity;
    
    int simulationSteps;
    int lastIndex;
    public:
        static PhysicsLevel &get()
        {
             static PhysicsLevel ins;
             return ins;
        }
        void load() {
             objects.clear();
             solvers.clear();
             add_solver(new PositionSolver());
             add_solver(new ElasticImpulseSolver());
             
             lastIndex = 0; 
             simulationSteps = 5;
             gravity = Vec3f(0.0f, -9.81f, 0.0f);
        }
        
        void add_object(Object *body) {
             body->index = lastIndex;
             
             objects.push_back(body);
             printf("%s\n", std::to_string(lastIndex).c_str());
             lastIndex++;
        }
        void remove_object(Object *body) {
             objects.erase(objects.begin() + body->index);
             //delete body;
        }
        
        void add_solver(Solver *solver) {
             solvers.push_back(solver);
        }
        
        void update(float timeTook) {
             float stepSize = timeTook / (float)simulationSteps;
             for (int i = 0; i < simulationSteps; i++) {
                  update_with_sub_steps(stepSize);
             }
        }
        void update_with_sub_steps(float timeTook) {
             resolve_collisions();
             
             // Reseting force and applying gravity
             for (auto &object : objects) {
                  RigidBody *body = (RigidBody*)object;
                  
                  body->force.set_zero();
                  if (!body->immovable) {
                       Vec3f acceleration = Vec3f(body->velocity).multiply(-1).add(gravity);
                       body->force.x += acceleration.x * body->mass;
                       body->force.y += acceleration.y * body->mass;
                       body->force.z += acceleration.z * body->mass;
                  }
             }
             
             // Euler's method
             for (auto &object : objects) {
                  RigidBody *body = (RigidBody*)object;
                  
                  body->velocity.x += (body->force.x * timeTook) / body->mass;
                  body->velocity.y += (body->force.y * timeTook) / body->mass;
                  body->velocity.z += (body->force.z * timeTook) / body->mass;
               
                  body->transform->position.x += body->velocity.x * timeTook;
                  body->transform->position.y += body->velocity.y * timeTook;
                  body->transform->position.z += body->velocity.z * timeTook;
             }
        }
        void resolve_collisions() {
             manifolds.clear();
             
             // Collision detection
             for (auto &object1 : objects) {
                  for (auto &object2 : objects) {
                       if (object1->index == object2->index) break;
                       if (!object1->collider || !object2->collider) continue;
                       
                       ManifoldPoints points = object1->collider->test(object1->transform, object2->collider, object2->transform);
                       if (points.collided) {
                            Manifold manifold = Manifold(object1, object2, points);
                            object1->colliding = &manifold;
                            object2->colliding = &manifold;
                            
                            object1->collidingNormal = points.normal;
                            object2->collidingNormal = points.normal;
                            manifolds.emplace_back(manifold);
                       }
                  }
             }
             for (auto &solver : solvers) {
                  solver->solve(manifolds);
             }
        }
        std::vector<Object*> &get_objects() { return objects; }
        Vec3f get_gravity() { return gravity; }
    private:
        PhysicsLevel() {}
        ~PhysicsLevel() {}
    public:
        PhysicsLevel(PhysicsLevel const&) = delete;
        void operator = (PhysicsLevel const&) = delete;    
};

class Level {
    public:
        static Level &get()
        {
            static Level ins;
            return ins;
        }
        void load() {
            objectShader = new Shader("object.vert", "object.frag");
            a = 0.0f;
        }
        
        void update(float timeTook) {
            a += timeTook;
            PhysicsLevel::get().update(timeTook);
            /*
            if (a > 1.0f) {
                 float size = rand() % 50 / 150.0f + 0.1f;
                 SphereObject *sphere = new SphereObject(size);
                 sphere->apply_velocity(rand() % 20 / 10.0f - 1.0f, 10.0f, rand() % 20 / 10.0f - 1.0f);
                 sphere->place(17.0f, 20.0f, 0.0f);
                 sphere->set_mass(size * 2.0f);
                 
                 sphere->set_material(rand() % 2 == 1 ? Materials::aluminium : Materials::gold);
                 
                 PhysicsLevel::get().add_object(sphere);
                 a = 0.0f;
            }
            */
        }
        
        void render(Camera *camera) {
            objectShader->use();
            objectShader->set_uniform_mat4("view", camera->get_view());
            objectShader->set_uniform_mat4("projection", camera->get_projection());
            objectShader->set_uniform_vec3f("viewPosition", camera->position.x, camera->position.y, camera->position.z);
        
         
            for (auto &object : PhysicsLevel::get().get_objects()) {
                 object->mesh->rotate(object->transform->rotation);
                 object->mesh->translate(object->transform->position);
                 
                 object->mesh->render(objectShader);
            }
        }
        void dispose() {
            for (auto &object : PhysicsLevel::get().get_objects()) {
                 object->mesh->dispose();
            }
            objectShader->clear();
        }
    private:
        Level() {}
        ~Level() {}
    public:
        Level(Level const&) = delete;
        void operator = (Level const&) = delete; 
        
    protected:
        Shader *objectShader; 
        float a = 0.0f;
};

class BallControls {
    public:
        BallControls(Camera *camera, SphereObject *ball) {
            this->camera = camera;
            this->ball = ball;
        }
        void handle_event(SDL_Event ev, float timeTook) {
            int w = 0, h = 0;
            SDL_GetWindowSize(windows, &w, &h);
            Vec2i click = this->get_mouse_position(ev);
           
            if (ev.type == SDL_MOUSEMOTION) {       
                float sensitivity = 0.1f;
               
                if (click.y < h / 2) {
                    camera->rotationX -= ev.motion.xrel * sensitivity * timeTook;
                    camera->rotationY += ev.motion.yrel * sensitivity * timeTook;
                }
                
                if (camera->rotationX > 2 * M_PI) camera->rotationX = 0;
                if (camera->rotationX < 0) camera->rotationX = 2 * M_PI;
               
                if (camera->rotationY > (89.0f / 180.0f * M_PI)) camera->rotationY = (89.0f / 180.0f * M_PI);
                if (camera->rotationY < -(89.0f / 180.0f * M_PI)) camera->rotationY = -(89.0f / 180.0f * M_PI);
              
                float s = 7.0f;
                Vec3f vel = Vec3f(cos(camera->rotationX),
                                  0.0f,
                                  sin(camera->rotationX));
                Vec3f v = vel.multiply(-s * timeTook);
                
                if (click.y > h / 2 && click.x < w / 2) {
                    ball->velocity.add(v);
                }
                else if (click.y > h / 2 && click.x > w / 2) {
                    Vec3f back = v.multiply(-1);
                    //camera->position.add(back);
                    ball->velocity.add(back);
                }
            }
            if (ev.type == SDL_MOUSEBUTTONUP) {
            //if (ev.type == SDL_MOUSEMOTION) {     
                if (click.y > h / 1.3f) {
                     jump();
                }
            }
        }
        void jump() {
            if (!ball->has_collided()) return;
            
            float strength = 5.0f;
            
            Vec3f normal = ball->collidingNormal;
            if (normal.dot(PhysicsLevel::get().get_gravity()) < 0) {
                 Vec3f offset = Vec3f(normal).multiply(0.1f);
                 Vec3f velocity = Vec3f(normal).multiply(strength);
                 
                 RigidBody *colliding = (RigidBody*) ball->colliding->object2;
                 //if (colliding->index == ball->index) colliding = (RigidBody*) ball->colliding->object1;
                 
                 ball->transform->position.add(offset);
                 ball->velocity.add(Vec3f(velocity).multiply(1 / ball->mass));
                 colliding->velocity.add(Vec3f(velocity).multiply(1 / colliding->mass));
                 
                 ball->colliding = 0;
            }
        }
        
        Vec2i get_mouse_position(SDL_Event event) {
            int dx = 0, dy = 0;
            SDL_GetMouseState(&dx, &dy);
            Vec2i result = { dx, dy };
           
            return result;
        }
    private:
        SphereObject *ball;
        Camera *camera;
};

class Game
{
  public:
    const char *displayName = "";
    virtual ~Game() {};
    virtual void init() {};
    virtual void load() {};
    
    virtual void handle_event(SDL_Event ev, float timeTook) {};

    virtual void update(float timeTook) {};
    virtual void dispose() {};
};

class ExampleRenderingEngine : public Game {
    Camera *camera;
    BallControls *controls;
    SphereObject *ball;
    public:  
       void init() override {
           displayName = "Example Rendering Engine";
       }
       void load() override {
           Level::get().load();
           Materials::load();
           PhysicsLevel::get().load();
           
           
           ball = new SphereObject(0.35f);
           ball->place(2.0f, 5.0f, 0.0f);
           ball->set_material(Materials::aluminium);
           PhysicsLevel::get().add_object(ball);
           
           camera = new Camera();
           controls = new BallControls(camera, ball);
           
           BoxObject *ground = new BoxObject(10.0f, 0.5f, 10.0f);
           ground->set_immovable(true);
           ground->place(0.0f, -1.0f, 0.0f);
           ground->set_texture("grass.png");
           ground->set_material(Materials::grass);
           
           PhysicsLevel::get().add_object(ground);
           
           for (auto &mesh : MeshGenerator::load_meshes_from_file("table.obj")) {
                ModelObject *model = new ModelObject(mesh);
                
                model->set_immovable(true);
                model->place_offset(0.0f, 0.0f, 0.0f);
                model->set_material(Materials::aluminium);
                
                PhysicsLevel::get().add_object(model);
           }
           
           for (auto &mesh : MeshGenerator::load_meshes_from_file("chair.obj")) {
                ModelObject *model = new ModelObject(mesh);
                
                model->set_immovable(true);
                model->place_offset(2.0f, 0.0f, 0.0f);
                model->set_material(Materials::aluminium);
                
                PhysicsLevel::get().add_object(model);
           }
           
       }
       void handle_event(SDL_Event ev, float timeTook) override {
           controls->handle_event(ev, timeTook);
       }
       void update(float timeTook) override {
           float distance = 4.0f;
           
           Level::get().update(timeTook);
           camera->lookingAt = Vec3f(ball->transform->position);
           camera->position = Vec3f(ball->transform->position).add(Vec3f(cos(camera->rotationX) * cos(camera->rotationY) * distance, sin(camera->rotationY) * distance, sin(camera->rotationX) * cos(camera->rotationY) * distance));
           camera->update();
           
           Level::get().render(camera);
       }
       
       void dispose() {  
           Level::get().dispose();
       }
};

int main()
{
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
	{
		fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
		return 1;
	}
    
	// We use OpenGL ES 3.0
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

	// We want at least 8 bits per color
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    
    
    ExampleRenderingEngine game;
    game.init();
    
	SDL_Window *window = SDL_CreateWindow(game.displayName, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL);
	if (window == NULL)
	{
		fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
		return 1;
	}
	
	// We will not actually need a context created, but we should create one
	SDL_GLContext context = SDL_GL_CreateContext(window);
    windows = window;
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    
    glDepthFunc(GL_LESS);
    glCullFace(GL_FRONT);
    
    game.load();    
    
	float then = 0.0f, delta = 0.0f;
    bool disabled = false;
    SDL_Event e;
    while (!disabled)
	{
		while (SDL_PollEvent(&e))
		{
			switch (e.type)
            {
                 case SDL_QUIT:
                      disabled = true;
                      break;
            }
            // Event-handling code
            game.handle_event(e, delta);
		}
		float now = SDL_GetTicks();
        delta = (now - then) * 1000 / SDL_GetPerformanceFrequency();
        then = now;
   
		// Drawing
		glClearColor(0.5f, 0.6f, 1.0f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    	
    	// Update and render to screen code
    	game.update(delta);
    	
		// Swap buffers
		SDL_GL_SwapWindow(window);
	}
	game.dispose();
	
    SDL_GL_DeleteContext(context);
    
	SDL_DestroyWindow(window);
	SDL_Quit();
	IMG_Quit();
	
	return 0;
}