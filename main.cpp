#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <vector>
#include <array>
#include <map>
#include <initializer_list>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_opengl.h>

//#include <GLES2/gl2.h>
#include <GLES3/gl3.h>

#include <ft2build.h>
#include FT_FREETYPE_H

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
     
    
     Vec2f nor() {
         return (*this) / len();
     }
     Vec2f perpendicular(int facing) {
         int j = facing >= 0 ? 1 : -1;
         float ax = x;
         float ay = y;
         
         x = j * ay;
         y = -j * ax;
        
         return *this;
     }
     Vec2f interpolate(const Vec2f &other, float progress) {
         Vec2f result;
         result.x = x + (other.x - x) * progress;
         result.y = y + (other.y - y) * progress;
        
         return result;
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
     
     Vec2f operator+(const Vec2f &other) {
         Vec2f result;
         result.x = x + other.x;
         result.y = y + other.y;
         return result;
     }
     Vec2f operator-(const Vec2f &other) {
         Vec2f result;
         result.x = x - other.x;
         result.y = y - other.y;
         return result;
     }
     Vec2f operator*(const Vec2f &other) {
         Vec2f result;
         result.x = x * other.x;
         result.y = y * other.y;
         return result;
     }
     Vec2f operator/(const Vec2f &other) {
         Vec2f result;
         result.x = x / other.x;
         result.y = y / other.y;
         return result;
     }
    
     Vec2f operator*(float scalar) {
         Vec2f result;
         result.x = x * scalar;
         result.y = y * scalar;
         return result;
     }
     Vec2f operator/(float scalar) {
         Vec2f result;
         result.x = x / scalar;
         result.y = y / scalar;
         return result;
     }
};

struct Vec3f {
    float x, y, z;
    Vec3f() {}
    Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    
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
        return (*this) / len();
    }
    
    // Component-wise clamping
    Vec3f clamp(const Vec3f &min, const Vec3f &max) {
        x = std::max(min.x, std::min(max.x, x));
        y = std::max(min.y, std::min(max.y, y));
        z = std::max(min.z, std::min(max.z, z));
        
        return *this;
    }
    Vec3f operator+(const Vec3f &other) {
        Vec3f result;
        result.x = x + other.x;
        result.y = y + other.y;
        result.z = z + other.z;
        return result;
    }
    Vec3f operator-(const Vec3f &other) {
        Vec3f result;
        result.x = x - other.x;
        result.y = y - other.y;
        result.z = z - other.z;
        return result;
    }
    Vec3f operator*(const Vec3f &other) {
        Vec3f result;
        result.x = x * other.x;
        result.y = y * other.y;
        result.z = z * other.z;
        return result;
    }
    Vec3f operator/(const Vec3f &other) {
        Vec3f result;
        result.x = x / other.x;
        result.y = y / other.y;
        result.z = z / other.z;
        return result;
    }
    
    Vec3f operator*(float scalar) {
        Vec3f result;
        result.x = x * scalar;
        result.y = y * scalar;
        result.z = z * scalar;
        return result;
    }
    Vec3f operator/(float scalar) {
        Vec3f result;
        result.x = x / scalar;
        result.y = y / scalar;
        result.z = z / scalar;
        return result;
    }
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
           Vec3f fwd = (cameraPosition - lookingAt).nor();                
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
       Mat4x4 operator*(const Mat4x4 &with) {
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

struct Tensor3x3 {
   public:
       float values[3][3] = {{ 0 }};
       Tensor3x3() {
             identity();
       }
       
       Vec3f multiply(const Vec3f &with) {
             Vec3f result;
             
             result.x = values[0][0] * with.x + values[1][0] * with.y + values[2][0] * with.z;
             result.y = values[0][1] * with.x + values[1][1] * with.y + values[2][1] * with.z;
             result.z = values[0][2] * with.x + values[1][2] * with.y + values[2][2] * with.z;
             
             return result;
       }
       
       Vec3f operator*(const Vec3f &with) {
             Vec3f result;
             
             result.x = values[0][0] * with.x + values[1][0] * with.y + values[2][0] * with.z;
             result.y = values[0][1] * with.x + values[1][1] * with.y + values[2][1] * with.z;
             result.z = values[0][2] * with.x + values[1][2] * with.y + values[2][2] * with.z;
             
             return result;
       }
       
       Tensor3x3 inverse() {
             float determinant = det();
             if (determinant == 0) {
                  printf("Couldn't divide by the zero determinant of a singular tensor.");
                  return identity();
             }
             
             float inverseDeterminant = 1 / determinant;
             float tmp[3][3];
             
             tmp[0][0] = values[1][1] * values[2][2] - values[2][1] * values[1][2];
             tmp[1][0] = values[2][0] * values[1][2] - values[1][0] * values[2][2];
             tmp[2][0] = values[1][0] * values[2][1] - values[2][0] * values[1][1];
              
             tmp[0][1] = values[2][1] * values[0][2] - values[0][1] * values[2][2];
             tmp[1][1] = values[0][0] * values[2][2] - values[2][0] * values[0][2];
             tmp[2][1] = values[2][0] * values[0][1] - values[0][0] * values[2][1];
             
             tmp[0][2] = values[0][1] * values[1][2] - values[1][1] * values[0][2];
             tmp[1][2] = values[1][0] * values[0][2] - values[0][0] * values[1][2];
             tmp[2][2] = values[0][0] * values[1][1] - values[1][0] * values[0][1];
             
             for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
                  values[i][j] = inverseDeterminant * tmp[i][j];
             }
             
             return *this;
       }
       Tensor3x3 identity() {
             for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
                  values[i][j] = 0.0f;
             }
             values[0][0] = 1.0f;
             values[1][1] = 1.0f;
             values[2][2] = 1.0f;
             
             return *this;
       }
       float det() {
             return values[0][0] * values[1][1] * values[2][2] + values[0][1] * values[1][2] * values[2][0] + values[0][2] * values[1][0] * values[2][1] -
                    values[0][0] * values[1][2] * values[2][1] - values[0][1] * values[1][0] * values[2][2] - values[0][2] * values[1][1] * values[2][0];
       }
};

struct Quaternion {
    float w, x, y, z;
    Quaternion() {}
    Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    
    // Roll - X rotation; Pitch - Y rotation; Yaw - Z rotation
    Quaternion from_euler(float roll, float pitch, float yaw) {
         float halfRoll = roll * 0.5f;
         float halfPitch = pitch * 0.5f;
         float halfYaw = yaw * 0.5f;
         
         // Abbreviations
         float cr = cos(halfRoll);
         float sr = sin(halfRoll);
         float cp = cos(halfPitch);
         float sp = sin(halfPitch);
         float cy = cos(halfYaw);
         float sy = sin(halfYaw);
         
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
         Quaternion temporary1 = Quaternion(reference.x, reference.y, reference.z, 0.0f), temporary2 = Quaternion(*this);
         temporary2 = temporary2.conjugate();
         temporary2 = temporary2 * temporary1 * (*this);
         
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
    
    Quaternion multiply_left(const Quaternion &other) {
         return multiply_left(other.x, other.y, other.z, other.w);
    }
    Quaternion operator*(const Quaternion &other) {
         float newX = other.w * x + other.x * w + other.y * z - other.z * y;
         float newY = other.w * y + other.y * w + other.z * x - other.x * z;
         float newZ = other.w * z + other.z * w + other.x * y - other.y * x;
         float newW = other.w * w - other.x * x - other.y * y - other.z * z;
         
         Quaternion r;
         r.x = newX;
         r.y = newY;
         r.z = newZ;
         r.w = newW;
         return r;
    }
};

struct Simplex {
    public:
         Simplex() : vertices({ Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f) }), currentSize(0) {}
         
         
         void push_front(const Vec3f &point) {
              vertices = { point, vertices[0], vertices[1], vertices[2] };
              currentSize = std::min(currentSize + 1, 4);
         }
         int size() { return currentSize; }
         auto begin() { return vertices.begin(); }
         auto end() { return vertices.end() - (4 - currentSize); }
         
         Simplex &operator=(std::initializer_list<Vec3f> from) {
              for (auto position = from.begin(); position != from.end(); position++) {
                   vertices[std::distance(from.begin(), position)] = *position;
              }
              
              currentSize = from.size();
              return *this;
         }
         Vec3f &operator[](int index) { return vertices[index]; }
    private:
         std::array<Vec3f, 4> vertices;
         int currentSize;
};

struct CharacterInfo {
    float advX, advY;
    
    float bitmapWidth, bitmapHeight;
    float bitmapLeft, bitmapTop;
    
    float offsetX = 0.0f;
};
class TextAtlas {
     public:
         TextAtlas(FT_Face font) {
              this->font = font;
              this->glyph = font->glyph;
              
              this->adjust_size();
         }
         void adjust_size() {
              int width = 0;
              int height = 0;
              
              // Load ASCII characters
              for (int i = 32; i < 128; i++) {
                   if (FT_Load_Char(font, i, FT_LOAD_RENDER)) {
                        printf("Couldn't load character %c.\n", i);
                        continue;
                   }
                   
                   width += glyph->bitmap.width;
                   if (glyph->bitmap.rows > height) {
                        height = glyph->bitmap.rows;
                   }
              }
              this->atlasWidth = width;
              this->atlasHeight = height;
         }
         
         void load() {
              this->add_empty_texture();
              
              int x = 0;
              for (int i = 32; i < 128; i++) {
                   if (FT_Load_Char(font, i, FT_LOAD_RENDER)) {
                        continue;
                   }
                   glTexSubImage2D(GL_TEXTURE_2D, 0, x, 0, glyph->bitmap.width, glyph->bitmap.rows, GL_RED, GL_UNSIGNED_BYTE, glyph->bitmap.buffer);
                   
                   CharacterInfo ch;
                   ch.advX = glyph->advance.x >> 6;
                   ch.advY = glyph->advance.y >> 6;
                   
                   ch.bitmapWidth = glyph->bitmap.width;
                   ch.bitmapHeight = glyph->bitmap.rows;
                   
                   ch.bitmapLeft = glyph->bitmap_left;
                   ch.bitmapTop = glyph->bitmap_top;
                   
                   ch.offsetX = (float) x / atlasWidth;
                   
                   characters[i] = ch;
                   
                   x += glyph->bitmap.width;
              }
         }
         void use() {
              glActiveTexture(GL_TEXTURE0);
              glBindTexture(GL_TEXTURE_2D, this->textureIndex);
         }
         
         void dispose() {
             glDeleteTextures(1, &textureIndex);
             FT_Done_Face(font);
         }
         
         std::array<CharacterInfo, 128> &get_characters() { return characters; }
         float get_width() { return atlasWidth; }
         float get_height() { return atlasHeight; }
         
     private:
         void add_empty_texture() {
              glActiveTexture(GL_TEXTURE0);
              glGenTextures(1, &textureIndex);
              
              glBindTexture(GL_TEXTURE_2D, textureIndex);
              glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
              
              glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, atlasWidth, atlasHeight, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
              
              
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                  
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
         }
     private:
         FT_Face font;
         FT_GlyphSlot glyph;
         GLuint textureIndex;
         
         int atlasWidth, atlasHeight;
         std::array<CharacterInfo, 128> characters;
};

class FreeType {
     public:
         static FreeType& get() {
              static FreeType ins;
              return ins;
         }
         
         void load() {
              errorHandler = FT_Init_FreeType(&library);
              if (errorHandler) {
                   throw std::runtime_error("FreeType library couldn't load.");
              }
              add_atlas("roboto", "/system/fonts/Roboto-Regular.ttf", 48);
         }
         
         void add_atlas(const char *name, const char *fileName, int height) {
              FT_Face font;
              
              errorHandler = FT_New_Face(library, fileName, 0, &font);
              if (errorHandler == FT_Err_Unknown_File_Format) {
                  throw std::runtime_error("The font file has an unknown format."); 
              } else if (errorHandler) {
                  throw std::runtime_error("Other error that occured when loading font.");
              }
              
              FT_Set_Pixel_Sizes(font, 0, height);
              TextAtlas *atlas = new TextAtlas(font);
              atlas->load();
              
              atlases[name] = atlas;
         }
         TextAtlas *find_atlas(const char *name) {
              return atlases[name];
         }
         
         void dispose() {
              for (auto &atlas : atlases) {
                   TextAtlas *second = atlas.second;
                   second->dispose();
              }
              FT_Done_FreeType(library);
         }
         
     private:
        FreeType() {}
        ~FreeType() {}
     public:
        FreeType(FreeType const&) = delete;
        void operator = (FreeType const&) = delete;
        
     private:
         FT_Library library;
         FT_Error errorHandler;
         std::map<const char*, TextAtlas*> atlases;
};


class Camera {
    public:
        Vec3f position;
        Vec3f lookingAt;
        Vec3f lookDirection;
        float rotationX;
        float rotationY;
        bool useRotation = true;
        
        Camera(float fov, float zNear, float zFar, float width, float height, bool perspective) {
            position = Vec3f(0.0f, 0.0f, 0.0f);
            rotationX = rotationY = 0.0f;
            lookingAt = Vec3f(1.0f, 0.0f, 0.0f);
            
            this->width = width;
            this->height = height;
            
            this->fov = fov;
            this->zNear = zNear;
            this->zFar = zFar;
            this->perspective = perspective;
        }
        
        Camera() : Camera(90.0f, 0.1f, 1000.0f, float(SCREEN_WIDTH), float(SCREEN_HEIGHT), true) {
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
            combined = viewMat * projMat;
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
            
            int clickX, clickY;
            SDL_GetMouseState(&clickX, &clickY);
            
            if (ev.type == SDL_MOUSEMOTION) {       
                float sensitivity = 0.1f;
               
                if (clickY < h / 2) {
                    camera->rotationX -= ev.motion.xrel * sensitivity * timeTook;
                    camera->rotationY -= ev.motion.yrel * sensitivity * timeTook;
                }
               
                if (camera->rotationX > 2 * M_PI) camera->rotationX = 0;
                if (camera->rotationX < 0) camera->rotationX = 2 * M_PI;
               
                if (camera->rotationY > (89.0f / 180.0f * M_PI)) camera->rotationY = (89.0f / 180.0f * M_PI);
                if (camera->rotationY < -(89.0f / 180.0f * M_PI)) camera->rotationY = -(89.0f / 180.0f * M_PI);
              
                float speed = 5.0f;
                Vec3f vel = Vec3f(cos(camera->rotationX) * cos(camera->rotationY),
                                  sin(camera->rotationY),
                                  sin(camera->rotationX) * cos(camera->rotationY));
                Vec3f v = vel * speed * timeTook;
                
                if (clickY > h / 2 && clickX < w / 2) {
                    camera->position = camera->position + v;
                }
                else if (clickY > h / 2 && clickX > w / 2) {
                    camera->position = camera->position - v;
                }
            }
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

struct OverlayVertex {
    Vec2f Position;
    Vec3f Color;
    Vec2f TextureCoords;
    
    OverlayVertex(float x = 0.0f, float y = 0.0f, float tx = 0.0f, float ty = 0.0f, Vec3f Color = Vec3f(1.0f, 1.0f, 1.0f)) : Position(x, y), TextureCoords(tx, ty), Color(Color) {}
};

struct MeshVertex {
    Vec3f Position;
    Vec3f Normal;
    Vec2f TextureCoords;
    
    MeshVertex(Vec3f Position, Vec3f Normal, Vec2f TextureCoords) : Position(Position), Normal(Normal), TextureCoords(TextureCoords) {}
    MeshVertex(float x = 0.0f, float y = 0.0f, float z = 0.0f, float nx = 0.0f, float ny = 0.0f, float nz = 0.0f, float tx = 0.0f, float ty = 0.0f) : Position(x, y, z), Normal(nx, ny, nz), TextureCoords(tx, ty) {}
};
struct BatchVertex {
    Vec3f Position;
    Vec3f Normal;
    Vec2f TextureCoords;
    Vec3f Color;
    
    BatchVertex() {}
    BatchVertex(Vec3f Position, Vec3f Normal, Vec2f TextureCoords) : Position(Position), Normal(Normal), TextureCoords(TextureCoords) {}
    BatchVertex(float x, float y, float z, float nx, float ny, float nz, float tx, float ty, Vec3f color) : Position(x, y, z), Normal(nx, ny, nz), TextureCoords(tx, ty), Color(color) {}
    BatchVertex(float x, float y, float z, Vec3f color) : Position(x, y, z), Normal(0, 0, 0), TextureCoords(0, 0), Color(color) {}
};
enum class TextureTypes {
    T2D = GL_TEXTURE_2D,
    TCube = GL_TEXTURE_CUBE_MAP,
};

template <TextureTypes T>
struct Texture {
    public:
         Texture() : repeating(true), smooth(false) {
              textureIndex = 0;
         }
         Texture(std::vector<std::string> fileNames) : Texture() {
              setup(fileNames);
         }
         void use() {
              if (textureIndex) {
                   GLenum type = static_cast<GLenum>(T);
                   
                   glActiveTexture(GL_TEXTURE0);
                   glBindTexture(type, textureIndex);
              }
         }
         void dispose() {
              if (textureIndex) glDeleteTextures(1, &textureIndex);
         }
    
         void setup(const std::vector<std::string> &fileNames) {
              GLenum type = static_cast<GLenum>(T);
              
              glGenTextures(1, &textureIndex);
              glBindTexture(type, textureIndex);
              if (type == GL_TEXTURE_2D) {
                       SDL_Surface *texture = load_surface(fileNames.at(0).c_str());
                       if (texture == NULL) {
                           printf("2D Texture couldn't load.\n");
                           return;
                       }
              
                       glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture->w, texture->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture->pixels);
                       if (repeating) {
                           glTexParameteri(type, GL_TEXTURE_WRAP_S, GL_REPEAT);
                           glTexParameteri(type, GL_TEXTURE_WRAP_T, GL_REPEAT);
                       } else {
                           glTexParameteri(type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                           glTexParameteri(type, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                       }
                       SDL_FreeSurface(texture);
              } else if (type == GL_TEXTURE_CUBE_MAP) {
                       for (int i = 0; i < fileNames.size(); i++) {
                            std::string faceName = fileNames.at(i);
                            SDL_Surface *texture = load_surface(faceName.c_str());
                            
                            if (texture == NULL) {
                                printf("Couldn't load the cubemap face/s.\n");
                                return;
                            }
                            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, texture->w, texture->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture->pixels);
                            
                            if (repeating) {
                                glTexParameteri(type, GL_TEXTURE_WRAP_S, GL_REPEAT);
                                glTexParameteri(type, GL_TEXTURE_WRAP_T, GL_REPEAT);
                                glTexParameteri(type, GL_TEXTURE_WRAP_R, GL_REPEAT);
                            } else {
                                glTexParameteri(type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                                glTexParameteri(type, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                                glTexParameteri(type, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
                            }
                            SDL_FreeSurface(texture);
                       }
              }
              
              glGenerateMipmap(type);
                   
              
              if (smooth) {
                   glTexParameteri(type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                   glTexParameteri(type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              } else {
                   glTexParameteri(type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                   glTexParameteri(type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
              }
              
              glBindTexture(type, 0);
         }
    protected:
         GLuint textureIndex;
         
         // Texture parameters
         bool repeating;
         bool smooth;
};

struct BatchType {
    GLenum renderType;
};
class OverlayBatch {
    public:
       BatchType type;
       OverlayBatch(int capacity, GLenum renderType, Shader *shader) {
           this->vertexCapacity = capacity;
           this->verticesUsed = 0;
           this->vbo = this->vao = 0;
         
           this->type.renderType = renderType;
           this->shader = shader;
           
           setup();
       }
       
       void setup() {
           glGenVertexArrays(1, &this->vao);
           glBindVertexArray(this->vao);
           
           glGenBuffers(1, &this->vbo);
           glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
           glBufferData(GL_ARRAY_BUFFER, this->vertexCapacity * sizeof(OverlayVertex), nullptr, GL_STREAM_DRAW); 
           
           GLint position = 0;
           glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*) offsetof(OverlayVertex, Position));
           glEnableVertexAttribArray(position);
           
           GLint color = 1;
           glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*) offsetof(OverlayVertex, Color));
           glEnableVertexAttribArray(color);
           
           GLint textureCoords = 2;
           glVertexAttribPointer(textureCoords, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*) offsetof(OverlayVertex, TextureCoords));
           glEnableVertexAttribArray(textureCoords);
           
           glBindVertexArray(0);
           glDisableVertexAttribArray(position);
           glDisableVertexAttribArray(color);
           glDisableVertexAttribArray(textureCoords);
           glBindBuffer(GL_ARRAY_BUFFER, 0);
       }
       
       void add(const std::vector<OverlayVertex> &vertices) {
           int extra = this->get_extra_vertices();
           if (vertices.size() + extra > vertexCapacity - verticesUsed) {
               return;
           }
           if (vertices.empty()) {
               return;
           }
           if (vertices.size() > vertexCapacity) {
               return;
           }
           
           glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
           if (extra > 0) {
               glBufferSubData(GL_ARRAY_BUFFER, (verticesUsed + 0) * sizeof(OverlayVertex), sizeof(OverlayVertex), &lastUsed);
               glBufferSubData(GL_ARRAY_BUFFER, (verticesUsed + 1) * sizeof(OverlayVertex), sizeof(OverlayVertex), &vertices[0]);
           }
           glBufferSubData(GL_ARRAY_BUFFER, verticesUsed * sizeof(OverlayVertex), vertices.size() * sizeof(OverlayVertex), &vertices[0]);
            
           glBindBuffer(GL_ARRAY_BUFFER, 0);
           verticesUsed += vertices.size() + extra;
           lastUsed = vertices.back();
       }
       
       void render() {
           if (verticesUsed == 0) {
               return;
           }
           glBindVertexArray(this->vao);
           glDrawArrays(this->type.renderType, 0, verticesUsed);
           
           this->verticesUsed = 0;
       }
       
       
       int get_extra_vertices() {
           bool mode = (this->type.renderType == GL_TRIANGLE_STRIP && verticesUsed > 0);
           return mode ? 2 : 0;
       }
       void dispose() {
           if (this->vbo) {
               glDeleteBuffers(1, &this->vbo);
           }
           if (this->vao) {
               glDeleteBuffers(1, &this->vao);
           }
       }
    protected:
       int vertexCapacity;
       int verticesUsed;
       OverlayVertex lastUsed;
       
       GLuint vbo;
       GLuint vao;
       Shader *shader;
};

class Batch {
    public:
       BatchType type;
       Batch(int capacity, GLenum renderType, Shader *shader) {
           this->vertexCapacity = capacity;
           this->verticesUsed = 0;
           this->vbo = this->vao = 0;
         
           this->type.renderType = renderType;
           this->shader = shader;
           
           setup();
       }
       
       void setup() {
           glGenVertexArrays(1, &this->vao);
           glBindVertexArray(this->vao);
           
           glGenBuffers(1, &this->vbo);
           glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
           glBufferData(GL_ARRAY_BUFFER, this->vertexCapacity * sizeof(BatchVertex), nullptr, GL_STREAM_DRAW); 
           
           GLint position = 0;
           glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, sizeof(BatchVertex), (void*) offsetof(BatchVertex, Position));
           glEnableVertexAttribArray(position);
           
           GLint normal = 1;
           glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, sizeof(BatchVertex), (void*) offsetof(BatchVertex, Normal));
           glEnableVertexAttribArray(normal);
           
           GLint textureCoords = 2;
           glVertexAttribPointer(textureCoords, 2, GL_FLOAT, GL_FALSE, sizeof(BatchVertex), (void*) offsetof(BatchVertex, TextureCoords));
           glEnableVertexAttribArray(textureCoords);
           
           GLint color = 3;
           glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, sizeof(BatchVertex), (void*) offsetof(BatchVertex, Color));
           glEnableVertexAttribArray(color);
           
           glBindVertexArray(0);
           glDisableVertexAttribArray(position);
           glDisableVertexAttribArray(normal);
           glDisableVertexAttribArray(textureCoords);
           glDisableVertexAttribArray(color);
           glBindBuffer(GL_ARRAY_BUFFER, 0);
       }
       
       void add(const std::vector<BatchVertex> &vertices) {
           int extra = this->get_extra_vertices();
           if (vertices.size() + extra > vertexCapacity - verticesUsed) {
               return;
           }
           if (vertices.empty()) {
               return;
           }
           if (vertices.size() > vertexCapacity) {
               return;
           }
           
           glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
           if (extra > 0) {
               glBufferSubData(GL_ARRAY_BUFFER, (verticesUsed + 0) * sizeof(BatchVertex), sizeof(BatchVertex), &lastUsed);
               glBufferSubData(GL_ARRAY_BUFFER, (verticesUsed + 1) * sizeof(BatchVertex), sizeof(BatchVertex), &vertices[0]);
           }
           glBufferSubData(GL_ARRAY_BUFFER, verticesUsed * sizeof(BatchVertex), vertices.size() * sizeof(BatchVertex), &vertices[0]);
            
           glBindBuffer(GL_ARRAY_BUFFER, 0);
           verticesUsed += vertices.size() + extra;
           lastUsed = vertices.back();
       }
       
       void render() {
           if (verticesUsed == 0) {
               return;
           }
           glBindVertexArray(this->vao);
           glDrawArrays(this->type.renderType, 0, verticesUsed);
           
           this->verticesUsed = 0;
       }
       
       
       int get_extra_vertices() {
           bool mode = (this->type.renderType == GL_TRIANGLE_STRIP && verticesUsed > 0);
           return mode ? 2 : 0;
       }
       void dispose() {
           if (this->vbo) {
               glDeleteBuffers(1, &this->vbo);
           }
           if (this->vao) {
               glDeleteBuffers(1, &this->vao);
           }
       }
    protected:
       int vertexCapacity;
       int verticesUsed;
       BatchVertex lastUsed;
       
       GLuint vbo;
       GLuint vao;
       Shader *shader;
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
              texture = new Texture<TextureTypes::T2D>();
              vbo = ibo = vao = 0;
         }
         void set_texture(const std::string &fileName) {
              texture->setup({ fileName });
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
              model = model * translation;
              model = model * rotation;
              model = model * scaling;
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
         Texture<TextureTypes::T2D> *texture;
         bool useTextures;
         bool useIndices = true;
         
         Mat4x4 scaling, rotation, translation;
         
         Material material;
         GLuint vbo, ibo;
         GLuint vao;
};

struct Skybox {
    public:
         Skybox(std::vector<std::string> fileNames) {
              std::vector<MeshVertex> vertices {
                   // Position
                   // Front
                   MeshVertex(-0.5, -0.5, 0.5),
                   MeshVertex(0.5, -0.5, 0.5),
                   MeshVertex(0.5, 0.5, 0.5),
                   MeshVertex(-0.5, 0.5, 0.5),
                   // Top
                   MeshVertex(-0.5, 0.5, 0.5),
                   MeshVertex(0.5, 0.5, 0.5),
                   MeshVertex(0.5, 0.5, -0.5),
                   MeshVertex(-0.5, 0.5, -0.5),
                   // Back
                   MeshVertex(0.5, -0.5, -0.5),
                   MeshVertex(-0.5, -0.5, -0.5),
                   MeshVertex(-0.5, 0.5, -0.5),
                   MeshVertex(0.5, 0.5, -0.5),
                   // Bottom
                   MeshVertex(-0.5, -0.5, -0.5),
                   MeshVertex(0.5, -0.5, -0.5),
                   MeshVertex(0.5, -0.5, 0.5),
                   MeshVertex(-0.5, -0.5, 0.5),
                   // Left
                   MeshVertex(-0.5, -0.5, -0.5),
                   MeshVertex(-0.5, -0.5, 0.5),
                   MeshVertex(-0.5, 0.5, 0.5),
                   MeshVertex(-0.5, 0.5, -0.5),
                   // Right
                   MeshVertex(0.5, -0.5, 0.5),
                   MeshVertex(0.5, -0.5, -0.5),
                   MeshVertex(0.5, 0.5, -0.5),  
                   MeshVertex(0.5, 0.5, 0.5)
              };
              
              for (auto &vertex : vertices) {
                   vertex.Position.x *= 2.0f;
                   vertex.Position.y *= 2.0f;
                   vertex.Position.z *= 2.0f;
                
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
         
              structure = MeshStructure(vertices, indices);
              texture = new Texture<TextureTypes::TCube>();
              texture->setup(fileNames);
              
              vbo = ibo = vao = 0;
              this->post_setup();
         }
         void render(Shader *shader) {
              if (!vao) return;
              
              glCullFace(GL_BACK);
              glDepthMask(GL_FALSE);
              texture->use();
              
              glBindVertexArray(vao);
              glDrawElements(GL_TRIANGLES, structure.indices.size(), GL_UNSIGNED_INT, 0);
              
              glBindVertexArray(0);
              glDepthMask(GL_TRUE);
              glCullFace(GL_FRONT);
         }
         void dispose() {
              if (vbo) glDeleteBuffers(1, &vbo);
              if (ibo) glDeleteBuffers(1, &ibo);
              if (vao) glDeleteVertexArrays(1, &vao);
              texture->dispose();
         }
    private:
         void post_setup() {
              if (!vao) glGenVertexArrays(1, &vao);
              glBindVertexArray(vao);
           
              if (!vbo) glGenBuffers(1, &vbo);
              if (!ibo) glGenBuffers(1, &ibo);
              glBindBuffer(GL_ARRAY_BUFFER, vbo);
              glBufferData(GL_ARRAY_BUFFER, structure.vertices.size() * sizeof(MeshVertex), structure.vertices.data(), GL_STATIC_DRAW); 
              
              glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
              glBufferData(GL_ELEMENT_ARRAY_BUFFER, structure.indices.size() * sizeof(GLuint), structure.indices.data(), GL_STATIC_DRAW);         
               
              // Position
              GLint position = 0;
              glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*) offsetof(MeshVertex, Position));
              glEnableVertexAttribArray(position);
              
              glBindVertexArray(0);
              glBindBuffer(GL_ARRAY_BUFFER, 0);
              glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
              
              glDisableVertexAttribArray(position);
         }
    protected:
         MeshStructure structure;
         Texture<TextureTypes::TCube> *texture;
         
         GLuint vbo, ibo;
         GLuint vao;
};

namespace Materials {
    Material aluminium, gold, copper,
             grass, wood, defaultMaterial;
    
    void load() {
         defaultMaterial = Material(Vec3f(1.0f, 1.0f, 1.0f), Vec3f(1.0f, 1.0f, 1.0f), Vec3f(0.0f, 0.0f, 0.0f), 0.1f);
         
         aluminium = Material(Vec3f(0.39f, 0.39f, 0.39f), Vec3f(0.504f, 0.504f, 0.504f), Vec3f(0.508f, 0.508f, 0.508f), 0.2f);
         gold = Material(Vec3f(0.39f, 0.34f, 0.22f), Vec3f(0.75f, 0.6f, 0.22f), Vec3f(0.62f, 0.55f, 0.36f), 0.2f);
         copper = Material(Vec3f(0.39f, 0.30f, 0.22f), Vec3f(0.73f, 0.30f, 0.11f), Vec3f(0.45f, 0.33f, 0.28f), 0.2f);
         
         grass = Material(Vec3f(0.7f, 0.7f, 0.7f), Vec3f(1.0f, 1.0f, 1.0f), Vec3f(0.0f, 0.0f, 0.0f));
         wood = Material(Vec3f(0.7f, 0.7f, 0.7f), Vec3f(1.0f, 1.0f, 1.0f), Vec3f(0.5f, 0.435f, 0.42f), 0.1f);
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
                       float phi = M_PI * 0.5f - i * latitudeSpacing;
                        
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
              vertex.Position.x *= width * 2;
              vertex.Position.y *= height * 2;
              vertex.Position.z *= depth * 2;
              
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
     
     void load_from_files(Mesh *source, const std::string &objName, const std::string &mtlName) {
         std::ifstream objRead(objName);
         std::vector<Vec3f> positions, normals;
         std::vector<Vec2f> textureCoordinates;
         std::vector<GLuint> vertexIndices, uvIndices, normalIndices;
         
         if (!objRead.is_open() || objRead.fail()) {
              printf("Couldn't open the .obj file.\n");
              return;
         }   
       
         std::string line; 
         while (std::getline(objRead, line)) {
              if (!line.compare("") || !line.compare(" ")) {
                  continue;
              }
              
              std::istringstream stream(line);
              std::string key;
              stream >> key;
              
              // Vertex position
              if (!key.compare("v")) {
                  Vec3f position;
                  stream >> position.x >> position.y >> position.z;
                  
                  positions.push_back(position);
              }
              // Vertex normal
              else if (!key.compare("vn")) {
                  Vec3f normal;
                  stream >> normal.x >> normal.y >> normal.z;
                  
                  normals.push_back(normal);
              }
              // Vertex texture coordinates
              else if (!key.compare("vt")) {
                  Vec2f coords;
                  stream >> coords.x >> coords.y;
                  
                  textureCoordinates.push_back(coords);
              }
              // Face indices
              else if (!key.compare("f")) {
                  std::string faceIndices[3];
                  stream >> faceIndices[0] >> faceIndices[1] >> faceIndices[2];
                  
                  for (int i = 0; i < 3; i++) {
                       GLuint indexPosition = 0, indexUV = 0, indexNormal = 0;
                       int scans = sscanf(faceIndices[i].c_str(), "%d/%d/%d", &indexPosition, &indexUV, &indexNormal);
                       if (scans != 3) {
                           sscanf(faceIndices[i].c_str(), "%d//%d", &indexPosition, &indexNormal);
                           indexUV = 0;
                       }
                       indexPosition--; indexUV--; indexNormal--;
                       
                       vertexIndices.push_back(indexPosition);
                       if (indexUV != -1) uvIndices.push_back(indexUV);
                       normalIndices.push_back(indexNormal);
                  }
              }
          }
          
          // Build model
          MeshStructure structure;
          for (int i = 0; i < vertexIndices.size(); i++) {
              structure.vertices.push_back(MeshVertex());
          }
          for (int i = 0; i < vertexIndices.size(); i++) {
              structure.vertices.at(i).Position = positions.at(vertexIndices.at(i));
          }
          for (int i = 0; i < uvIndices.size(); i++) {
              structure.vertices.at(i).TextureCoords = textureCoordinates.at(uvIndices.at(i));
          }
          for (int i = 0; i < normalIndices.size(); i++) {
              structure.vertices.at(i).Normal = normals.at(normalIndices.at(i));
          }
              
          source->unuse_indices();
          source->set_structure(structure);
     } 
     
     
     // Loads meshes from a single model, based on the "g" qualifier
     std::vector<MeshStructure> load_meshes_from_file(const std::string &objName) {
         std::vector<MeshStructure> result;
         std::ifstream objRead(objName);
         
         std::vector<Vec3f> positions, normals;
         std::vector<Vec2f> textureCoordinates;
         std::vector<GLuint> vertexIndices, uvIndices, normalIndices;
         
         if (!objRead.is_open() || objRead.fail()) {
              printf("Couldn't open the .obj file.\n");
              return result;
         }   
       
         std::string line; 
         while (std::getline(objRead, line)) {
              if (!line.compare("") || !line.compare(" ")) {
                  continue;
              }
              
              std::istringstream stream(line);
              std::string key;
              stream >> key;
              
              // Vertex position
              if (!key.compare("v")) {
                  Vec3f position;
                  stream >> position.x >> position.y >> position.z;
                  
                  positions.push_back(position);
              }
              // Vertex normal
              else if (!key.compare("vn")) {
                  Vec3f normal;
                  stream >> normal.x >> normal.y >> normal.z;
                  
                  normals.push_back(normal);
              }
              // Vertex texture coordinates
              else if (!key.compare("vt")) {
                  Vec2f coords;
                  stream >> coords.x >> coords.y;
                  
                  textureCoordinates.push_back(coords);
              }
              // Face indices
              else if (!key.compare("f")) {
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
              if (!key.compare("g")) {
                  MeshStructure structure;
                  for (int i = 0; i < vertexIndices.size(); i++) {
                       GLuint indexPosition = vertexIndices.at(i);
                       GLuint indexUV = uvIndices.at(i);
                       GLuint indexNormal = normalIndices.at(i);
              
                       MeshVertex vertex = MeshVertex(positions.at(indexPosition), normals.at(indexNormal), textureCoordinates.at(indexUV));
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
              
              MeshVertex vertex = MeshVertex(positions.at(indexPosition), normals.at(indexNormal), textureCoordinates.at(indexUV));
              structure.vertices.push_back(vertex);
         }
         if (!vertexIndices.empty()) {
              result.push_back(structure);
         } 
            
         return result;
     }
     
     std::vector<Vec3f> load_vertices(const std::string &objName) {
         std::ifstream objRead(objName);
         std::vector<Vec3f> positions;
         
         if (!objRead.is_open() || objRead.fail()) {
              printf("Couldn't open the .obj file.\n");
              return positions;
         }   
       
         std::string line; 
         while (std::getline(objRead, line)) {
              if (!line.compare("") || !line.compare(" ")) {
                  continue;
              }
              
              std::istringstream stream(line);
              std::string key;
              stream >> key;
              
              // Vertex position
              if (!key.compare("v")) {
                  Vec3f position;
                  stream >> position.x >> position.y >> position.z;
                  
                  positions.push_back(position);
              }
          }
          return positions;
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

namespace UIRenderer {
     Shader *overlayShader;
     OverlayBatch *batch;
     TextAtlas *textAtlas;
     void load() {
           overlayShader = new Shader("resources/shaders/overlay.vert", "resources/shaders/overlay.frag"); 
           batch = new OverlayBatch(4096, GL_TRIANGLES, overlayShader);
           
           textAtlas = FreeType::get().find_atlas("roboto");
     }
     void render() {
           glDisable(GL_DEPTH_TEST);
           overlayShader->use();
           
           textAtlas->use();
           batch->render();
           
           glEnable(GL_DEPTH_TEST);
     }
     
     void dispose() {
           overlayShader->clear();
           batch->dispose();
     }
     
     void draw_string_itself(const std::string &text, float x, float y, float sclX, float sclY, const Vec3f &color) {
           std::vector<OverlayVertex> vertices;
             
           float px = x;
           float py = y;
           std::string::const_iterator iterator;
           for (iterator = text.begin(); iterator != text.end(); iterator++) {
                CharacterInfo ch = textAtlas->get_characters().at(*iterator);
                   
                float x2 = px + ch.bitmapLeft * sclX;
                float y2 = -py - ch.bitmapTop * sclY;
                float width = ch.bitmapWidth * sclX;
                float height = ch.bitmapHeight * sclY;
                   
                px += ch.advX * sclX;
                py += ch.advY * sclY;
                   
                   
                vertices.push_back(OverlayVertex(x2, -y2, ch.offsetX, 0, color));
                vertices.push_back(OverlayVertex(x2 + width, -y2, ch.offsetX + ch.bitmapWidth / textAtlas->get_width(), 0, color));
                vertices.push_back(OverlayVertex(x2, -y2 - height, ch.offsetX, ch.bitmapHeight / textAtlas->get_height(), color));
                   
                vertices.push_back(OverlayVertex(x2 + width, -y2, ch.offsetX + ch.bitmapWidth / textAtlas->get_width(), 0, color));
                vertices.push_back(OverlayVertex(x2 + width, -y2 - height, ch.offsetX + ch.bitmapWidth / textAtlas->get_width(), ch.bitmapHeight / textAtlas->get_height(), color));
                vertices.push_back(OverlayVertex(x2, -y2 - height, ch.offsetX, ch.bitmapHeight / textAtlas->get_height(), color));
           }
              
           batch->add(vertices);
     }
     void draw_string(const std::string &text, float x, float y, float scaleX = 1.0f, float scaleY = 1.0f, const Vec3f &color = Vec3f(1.0f, 1.0f, 1.0f)) {
           float sclX = scaleX * 0.002f;
           float sclY = scaleY * 0.002f;
           
           draw_string_itself(text, x, y, sclX, sclY, color);
     }
     void draw_string_centered(const std::string &text, float x, float y, float scaleX = 1.0f, float scaleY = 1.0f, const Vec3f &color = Vec3f(1.0f, 1.0f, 1.0f)) {
           float sclX = scaleX * 0.002f;
           float sclY = scaleY * 0.002f;
           
           float w = 0.0f;
           float h = 0.0f;
           std::string::const_iterator iterator;
           for (iterator = text.begin(); iterator != text.end(); iterator++) {
                CharacterInfo ch = textAtlas->get_characters().at(*iterator);
                
                w += ch.advX;
                
                if (ch.bitmapHeight > h) {
                    h = ch.bitmapHeight;
                }
           }
           if (!text.empty()) {
                CharacterInfo ch = textAtlas->get_characters().at(*text.end());
                w -= (ch.advX - (ch.bitmapLeft + ch.bitmapWidth));
           }
           
           w *= sclX;
           h *= sclY;
           
           
           float px = x - w / 2.0f;
           float py = y - h / 2.0f;
           draw_string_itself(text, px, py, sclX, sclY, color);
     }
};
namespace BatchRenderer {
     Shader *shader;
     Batch *lineBatch;
     std::vector<Vec3f> sphereVertices;
     void load() {
           shader = new Shader("resources/shaders/batch.vert", "resources/shaders/batch.frag");      
           lineBatch = new Batch(40960, GL_LINES, shader);
           
           MeshStructure sphere = MeshGenerator::get_sphere_mesh(8, 8);
           for (auto &index : sphere.indices) {
                Vec3f position = sphere.vertices.at(index).Position;
                
                sphereVertices.push_back(position);
           }
     }
     void render(Camera *camera) {
           Mat4x4 model;
           
           shader->use();
           shader->set_uniform_mat4("model", model);
           shader->set_uniform_mat4("view", camera->get_view());
           shader->set_uniform_mat4("projection", camera->get_projection());
            
           lineBatch->render();
     }
   
     void dispose() {
           lineBatch->dispose();
     }
     
     void draw_spring(float x1, float y1, float z1, float x2, float y2, float z2, const Vec3f &color = Vec3f(1.0f, 1.0f, 1.0f)) {
           std::vector<BatchVertex> vertices = {
                BatchVertex(x1, y1, z1, color),
                BatchVertex(x2, y2, z2, color)
           };
           
           lineBatch->add(vertices);
     }
     void draw_box_outline(float x, float y, float z, float width, float height, float depth, Quaternion rotation, const Vec3f &color = Vec3f(1.0f, 1.0f, 1.0f)) {
           std::vector<BatchVertex> vertices = {
                BatchVertex(-1.0, -1.0, -1.0, color),
                BatchVertex(1.0, -1.0, -1.0, color),
                BatchVertex(-1.0, 1.0, -1.0, color),
                BatchVertex(1.0, 1.0, -1.0, color),
            
                BatchVertex(-1.0, -1.0, 1.0, color),
                BatchVertex(1.0, -1.0, 1.0, color),
                BatchVertex(-1.0, 1.0, 1.0, color),
                BatchVertex(1.0, 1.0, 1.0, color),
            
                BatchVertex(-1.0, -1.0, -1.0, color),
                BatchVertex(-1.0, 1.0, -1.0, color),
                BatchVertex(1.0, -1.0, -1.0, color),
                BatchVertex(1.0, 1.0, -1.0, color),
            
                BatchVertex(-1.0, -1.0, 1.0, color),
                BatchVertex(-1.0, 1.0, 1.0, color),
                BatchVertex(1.0, 1.0, 1.0, color),
                BatchVertex(1.0, -1.0, 1.0, color),
            
                BatchVertex(-1.0, -1.0, -1.0, color),
                BatchVertex(-1.0, -1.0, 1.0, color),
                BatchVertex(1.0, -1.0, -1.0, color),
                BatchVertex(1.0, -1.0, 1.0, color),
            
                BatchVertex(-1.0, 1.0, -1.0, color),
                BatchVertex(-1.0, 1.0, 1.0, color),
                BatchVertex(1.0, 1.0, -1.0, color),
                BatchVertex(1.0, 1.0, 1.0, color)
           };
           for (auto &vertex : vertices) {
                vertex.Position.x *= width;
                vertex.Position.y *= height;
                vertex.Position.z *= depth;
                
                vertex.Position = rotation.transform(vertex.Position);
                
                vertex.Position.x += x;
                vertex.Position.y += y;
                vertex.Position.z += z;
           }
           
           lineBatch->add(vertices);
     }
     void draw_sphere_outline(float x, float y, float z, float radius, const Vec3f &color = Vec3f(1.0f, 1.0f, 1.0f)) {
           std::vector<BatchVertex> vertices;
           
           for (auto &position : sphereVertices) {
                Vec3f p = position * radius;
                p.x += x;
                p.y += y;
                p.z += z;
                
                BatchVertex vertex = BatchVertex(p.x, p.y, p.z, color);
                vertices.push_back(vertex);
           }
           
           lineBatch->add(vertices); 
     }
                       
};

// ----- Physics -----
static struct Collider;
static struct SphereCollider;
static struct BoxCollider;
static struct MeshCollider;
static struct ColliderGroup;

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


struct Collider {
     public:
         float boundingRadius = 0.0f;
         
         // Collider-collider
         virtual ManifoldPoints test(Transform *transform, Collider *collider, Transform *otherTransform) = 0;
         // Collider-sphere
         virtual ManifoldPoints test(Transform *transform, SphereCollider *sphereCollider, Transform *otherTransform) = 0;
         // Collider-box
         virtual ManifoldPoints test(Transform *transform, BoxCollider *boxCollider, Transform *otherTransform) = 0;
         // Collider-mesh
         virtual ManifoldPoints test(Transform *transform, MeshCollider *meshCollider, Transform *otherTransform) = 0;
     
         // Assuming an unit direction
         virtual Vec3f support_point(Transform *transform, Vec3f direction) = 0;
         // Bounding sphere radius
         virtual float compute_bounding() = 0;
};

// Adapted from https://blog.winter.dev/#articles
namespace GJK {
      float MAX_VALUE = 10000.0f;
      float COLLISION_THRESHOLD = 0.001f;
      int MAX_ITERATION_COUNT = 10;
      
      struct PolytopeResult {
            Vec3f vector;
            float distance;
            
            PolytopeResult() {}
            PolytopeResult(Vec3f vector, float distance) : vector(vector), distance(distance) {}
      };
      
      // ----- GJK -----
      // Returns a support point on the Minkovski difference of two colliders
      Vec3f get_support_point_minkovski(Collider *collider, Collider *other, Transform *transform, Transform *otherTransform, Vec3f direction) {
            Vec3f support1 = collider->support_point(transform, direction);
            Vec3f support2 = other->support_point(otherTransform, direction * -1);
            
            return support1 - support2;
      }
      
      bool same_direction(const Vec3f &direction, const Vec3f &AtoOrigin) {
            Vec3f dir = direction;
            return dir.dot(AtoOrigin) > 0;
      }
      
      bool test_line(Simplex &points, Vec3f &direction) {
            Vec3f a = points[0], b = points[1];
            
            Vec3f gradient = b - a;
            Vec3f AtoOrigin = a * -1;
            
            if (same_direction(gradient, AtoOrigin)) {
                 direction = gradient.crs(AtoOrigin).crs(gradient);
            } else {
                 points = { a };
                 direction = AtoOrigin;
            }
            
            return false;
      }
      bool test_triangle(Simplex &points, Vec3f &direction) {
            Vec3f a = points[0], b = points[1], c = points[2];
            
            Vec3f ab = b - a,
                  ac = c - a,
                  AtoOrigin = a * -1;
            Vec3f abc = ab.crs(ac);
            
            if (same_direction(abc.crs(ac), AtoOrigin)) {
                 if (same_direction(ac, AtoOrigin)) {
                      points = { a, c };
                      direction = ac.crs(AtoOrigin).crs(ac);
                 } else return test_line(points = { a, b }, direction);
            } else {
                 if (same_direction(ab.crs(abc), AtoOrigin)) return test_line(points = { a, b }, direction);
                 else {
                      if (same_direction(abc, AtoOrigin)) direction = abc;
                      else {
                           points = { a, c, b };
                           direction = abc * -1;
                      }
                 }
            }
            
            return false;
      }
      bool test_tetrahedron(Simplex &points, Vec3f &direction) {
            Vec3f a = points[0], b = points[1], c = points[2], d = points[3];
            
            Vec3f ab = b - a, 
                  ac = c - a,
                  ad = d - a,
                  AtoOrigin = a * -1;
                  
            Vec3f abc = ab.crs(ac),
                  acd = ac.crs(ad),
                  adb = ad.crs(ab);
            
            if (same_direction(abc, AtoOrigin)) return test_triangle(points = { a, b, c }, direction);
            if (same_direction(acd, AtoOrigin)) return test_triangle(points = { a, c, d }, direction);
            if (same_direction(adb, AtoOrigin)) return test_triangle(points = { a, d, b }, direction);
            
            return true;
      }
      
      bool next_simplex(Simplex &points, Vec3f &direction) {
            switch (points.size()) {
                 case 2: return test_line(points, direction);
                 case 3: return test_triangle(points, direction);
                 case 4: return test_tetrahedron(points, direction);
            }
            return false;
      }
      // Whether or not the Minkovski difference contains the origin
      bool contains_origin(Simplex &points, Collider *collider, Collider *other, Transform *transform, Transform *otherTransform) {
            Vec3f support = get_support_point_minkovski(collider, other, transform, otherTransform, Vec3f(1.0f, 0.0f, 0.0f));
            Vec3f direction = support * -1;
          
            points.push_front(support);
            
            while (true) {
                  support = get_support_point_minkovski(collider, other, transform, otherTransform, direction);
                  if (support.dot(direction) <= 0) return false;
                  
                  points.push_front(support);
                  if (next_simplex(points, direction)) return true;
            }
      }
      
      // ----- EPA -----
      std::pair<std::vector<PolytopeResult>, GLuint> get_polytope_normals(const std::vector<Vec3f> &polytope, const std::vector<GLuint> &faces) {
            std::vector<PolytopeResult> normals;
            GLuint minimumFace = 0;
            float minimumDistance = MAX_VALUE;
            
            for (int i = 0; i < faces.size(); i += 3) {
                 Vec3f a = polytope[faces[i]];
                 Vec3f b = polytope[faces[i + 1]];
                 Vec3f c = polytope[faces[i + 2]];
                 
                 Vec3f normal = (b - a).crs(c - a).nor();
                 float dot = normal.dot(a);
                 
                 if (dot < 0) {
                      normal = normal * -1;
                      dot = dot * -1;
                 }
                 
                 normals.emplace_back(PolytopeResult(normal, dot));
                 if (dot < minimumDistance) {
                      minimumFace = i / 3;
                      minimumDistance = dot;
                 }
            }
            return { normals, minimumFace };
      }
      
      void add_unique_edge(std::vector<std::pair<GLuint, GLuint>> &edges, const std::vector<GLuint> &faces, GLuint a, GLuint b) {
            auto reverse = std::find(edges.begin(), edges.end(), std::make_pair(faces[b], faces[a]));
            
            if (reverse != edges.end()) {
                 edges.erase(reverse);
            } else {
                 edges.emplace_back(faces[a], faces[b]);
            }
      }
      
      ManifoldPoints expanding_polytope(Simplex &simplex, Collider *collider, Collider *other, Transform *transform, Transform *otherTransform) {
            int iterations = 0;
            
            std::vector<Vec3f> polytope(simplex.begin(), simplex.end());
            
            std::vector<GLuint> faces = {
                 0, 1, 2,
                 0, 3, 1,
                 0, 2, 3,
                 1, 3, 2
            };
            auto [normals, minimumFace] = get_polytope_normals(polytope, faces);
            
            Vec3f minimumNormal;
            float minimumDistance = MAX_VALUE;
            
            while (minimumDistance == MAX_VALUE) {
                  minimumNormal = normals[minimumFace].vector;
                  minimumDistance = normals[minimumFace].distance;
                  
                  if (iterations++ > MAX_ITERATION_COUNT) {
                       break;
                  }
                  
                  Vec3f support = get_support_point_minkovski(collider, other, transform, otherTransform, minimumNormal);
                  float dot = support.dot(minimumNormal);
                  
                  if (abs(dot - minimumDistance) > COLLISION_THRESHOLD) {
                       minimumDistance = MAX_VALUE;
                       
                       std::vector<std::pair<GLuint, GLuint>> uniqueEdges;
                       for (int i = 0; i < normals.size(); i++) {
                            if (!same_direction(normals[i].vector, support)) continue;
                            
                            int f = i * 3;
                            add_unique_edge(uniqueEdges, faces, f, f + 1);
                            add_unique_edge(uniqueEdges, faces, f + 1, f + 2);
                            add_unique_edge(uniqueEdges, faces, f + 2, f);
                            
                            faces[f + 2] = faces.back(); faces.pop_back();
                            faces[f + 1] = faces.back(); faces.pop_back();
                            faces[f] = faces.back(); faces.pop_back();
                            
                            normals[i] = normals.back(); normals.pop_back();
                            
                            i--;
                       }
                       
                       if (uniqueEdges.size() == 0) {
                            break;
                       }
                       
                       std::vector<GLuint> newFaces;
                       for (auto [edgeIndex1, edgeIndex2] : uniqueEdges) {
                            newFaces.push_back(edgeIndex1);
                            newFaces.push_back(edgeIndex2);
                            newFaces.push_back(polytope.size());
                       }
                       polytope.push_back(support);
                       
                       
                       auto [newNormals, newMinimumFace] = get_polytope_normals(polytope, newFaces);
                       float newMinimumDistance = MAX_VALUE;
                       
                       for (int i = 0; i < normals.size(); i++) {
                            if (normals[i].distance < newMinimumDistance) {
                                 newMinimumDistance = normals[i].distance;
                                 minimumFace = i;
                            }
                       }
                       
                       if (newNormals[newMinimumFace].distance < newMinimumDistance) {
                            minimumFace = newMinimumFace + normals.size();
                       }
                       
                       faces.insert(faces.end(), newFaces.begin(), newFaces.end());
                       normals.insert(normals.end(), newNormals.begin(), newNormals.end());
                  }
            }
            if (minimumDistance == MAX_VALUE) {
                  return ManifoldPoints();
            }
            
            ManifoldPoints result;
            result.normal = minimumNormal * -1;
            result.depth = minimumDistance + COLLISION_THRESHOLD;
            result.collided = true;
            result.AtoB = otherTransform->position;
            result.BtoA = transform->position;
            
            return result;
      }
};

namespace CollisionDetection {
      ManifoldPoints sphere_sphere(SphereCollider *sphere, Transform *transform, SphereCollider *other, Transform *otherTransform);
      ManifoldPoints sphere_AABB(SphereCollider *sphere, Transform *transform, BoxCollider *other, Transform *otherTransform);
      ManifoldPoints sphere_OBB(SphereCollider *sphere, Transform *transform, BoxCollider *other, Transform *otherTransform);
      ManifoldPoints GJK(Collider *collider, Transform *transform, Collider *other, Transform *otherTransform);
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
     // Sphere-Mesh
     ManifoldPoints test(Transform *transform, MeshCollider *meshCollider, Transform *otherTransform) override {
           return CollisionDetection::GJK(this, transform, (Collider*)meshCollider, otherTransform);
     }
     
     Vec3f support_point(Transform *transform, Vec3f direction) override {
           return transform->position + (direction.nor()) * radius;
     }
     float compute_bounding() override {
           return radius;
     }
};

struct BoxCollider : Collider {
     float width, height, depth;
     std::vector<Vec3f> vertices;
     
     BoxCollider() {}
     BoxCollider(float width, float height, float depth) : width(width), height(height), depth(depth) {
          this->vertices = {
               Vec3f(-width, -height, -depth),
               Vec3f(width, -height, -depth),
               Vec3f(width, -height, depth),
               Vec3f(-width, -height, depth),
               
               Vec3f(-width, height, -depth),
               Vec3f(width, height, -depth),
               Vec3f(width, height, depth),
               Vec3f(-width, height, depth),
          };
     }
     BoxCollider(Vec3f size) : BoxCollider(size.x, size.y, size.z) {
          
     }
     
     // Box-collider
     ManifoldPoints test(Transform *transform, Collider *collider, Transform *otherTransform) override { 
           return collider->test(otherTransform, this, transform); 
     }
     // Box-sphere
     ManifoldPoints test(Transform *transform, SphereCollider *sphereCollider, Transform *otherTransform) override { 
           // Reuse sphere code
           ManifoldPoints points = sphereCollider->test(otherTransform, this, transform);
           
           // Swap the points, so that the collision might not break
           Vec3f temporary = points.AtoB;
           points.AtoB = points.BtoA;
           points.BtoA = temporary;
           points.normal = points.normal * -1;
           
           return points;
     }
     // Box-box
     ManifoldPoints test(Transform *transform, BoxCollider *boxCollider, Transform *otherTransform) override { 
           //return CollisionDetection::GJK(this, transform, (Collider*)boxCollider, otherTransform);
           return ManifoldPoints();
     }
     // Box-Mesh
     ManifoldPoints test(Transform *transform, MeshCollider *meshCollider, Transform *otherTransform) override {
           return CollisionDetection::GJK(this, transform, (Collider*)meshCollider, otherTransform);
     }
     
     Vec3f support_point(Transform *transform, Vec3f direction) override {
           Vec3f result = Vec3f(0.0f, 0.0f, 0.0f);
           float maxDot = -10000.0f;
            
           for (auto &vertex : vertices) {
                 Vec3f point = vertex;
                 point = transform->rotation.transform(point);
                 point = point + transform->position;
                 
                 float dot = direction.dot(point);
                 if (dot > maxDot) {
                      result = point;
                      maxDot = dot;
                 }
           }
           
           return result;
     }
     float compute_bounding() override {
           return sqrt(width*width + height*height + depth*depth);
     }
        
     bool contains_point(Transform *transform, const Vec3f &point) {
           return (point.x >= transform->position.x - width && point.x <= transform->position.x + width) &&
                  (point.y >= transform->position.y - height && point.y <= transform->position.y + height) &&
                  (point.z >= transform->position.z - depth && point.z <= transform->position.z + depth);
     }
};
struct MeshCollider : Collider {
     std::vector<Vec3f> vertices;
     
     MeshCollider() {}
     MeshCollider(const std::vector<Vec3f> &vertices) : vertices(vertices) {}
     
     // Mesh-collider
     ManifoldPoints test(Transform *transform, Collider *collider, Transform *otherTransform) override {
           return collider->test(otherTransform, this, transform); 
     }
     
     // Mesh-sphere
     ManifoldPoints test(Transform *transform, SphereCollider *sphereCollider, Transform *otherTransform) override {
           // Reuse sphere code
           ManifoldPoints points = sphereCollider->test(otherTransform, this, transform);
           
           // Swap the points, so that the collision might not break
           Vec3f temporary = points.AtoB;
           points.AtoB = points.BtoA;
           points.BtoA = temporary;
           points.normal = points.normal * -1;
           
           return points;
     }
     
     // Mesh-box
     ManifoldPoints test(Transform *transform, BoxCollider *boxCollider, Transform *otherTransform) override {
           // Reuse box code
           ManifoldPoints points = boxCollider->test(otherTransform, this, transform);
           
           // Swap the points, so that the collision might not break
           Vec3f temporary = points.AtoB;
           points.AtoB = points.BtoA;
           points.BtoA = temporary;
           points.normal = points.normal * -1;
           
           return points;
     }
     
     // Mesh-mesh
     ManifoldPoints test(Transform *transform, MeshCollider *meshCollider, Transform *otherTransform) override {
           return CollisionDetection::GJK(this, transform, (Collider*)meshCollider, otherTransform);
     }
     
     Vec3f support_point(Transform *transform, Vec3f direction) override {
           Vec3f result = Vec3f(0.0f, 0.0f, 0.0f);
           float maxDot = -10000.0f;
            
           for (auto &vertex : vertices) {
                 Vec3f point = vertex;
                 point = transform->rotation.transform(point);
                 point = point + transform->position;
           
                 float dot = direction.dot(point);
                 if (dot > maxDot) {
                      result = point;
                      maxDot = dot;
                 }
           }
           
           return result;
     }
     float compute_bounding() override {
           float maxRadius = -10000.0f;
           
           for (auto &vertex : vertices) {
                 float distance = vertex.len2();
                 if (distance > maxRadius) {
                      maxRadius = distance;
                 }
           }
           maxRadius = sqrt(maxRadius);
           
           return maxRadius;
     }
};
struct ColliderGroup : Collider {
     std::vector<std::pair<Collider*, Transform*>> colliders;
     
     ColliderGroup() {}
     
     // Group-collider
     ManifoldPoints test(Transform *transform, Collider *collider, Transform *otherTransform) override {
          return ManifoldPoints();
     }
     
     // Group-sphere
     ManifoldPoints test(Transform *transform, SphereCollider *sphereCollider, Transform *otherTransform) override {
          return ManifoldPoints();
     }
     
     // Group-box
     ManifoldPoints test(Transform *transform, BoxCollider *boxCollider, Transform *otherTransform) override {
          return ManifoldPoints();
     }
     // Group-mesh
     ManifoldPoints test(Transform *transform, MeshCollider *meshCollider, Transform *otherTransform) override {
          return ManifoldPoints();
     }
     
     Vec3f support_point(Transform *transform, Vec3f direction) override {
          return Vec3f(0.0f, 0.0f, 0.0f);
     }
     float compute_bounding() override {
          float maxDistance = -10000.0f;
          Collider *furthest = nullptr;
          
          for (auto &pair : colliders) {
               float distance = pair.second->position.len2();
               if (distance > maxDistance) {
                    maxDistance = distance;
                    furthest = pair.first;
               }
          }
          maxDistance = sqrt(maxDistance);
          
          float radius = furthest->compute_bounding();
          float result = maxDistance + radius;
          
          return result;
     }
     
     void append(Collider *collider, Transform *location) {
          colliders.emplace_back(std::make_pair(collider, location));
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
          Vec3f AtoB = (otherPosition - position).nor();
          Vec3f BtoA = AtoB * -1;
          AtoB = AtoB * radiusOther + position;
          BtoA = BtoA * radius + otherPosition;
          
          Vec3f direction = BtoA - AtoB;
          Vec3f normal = Vec3f(direction).nor();
          return ManifoldPoints(AtoB, BtoA, normal, direction.len());
     }
     ManifoldPoints sphere_AABB(SphereCollider *sphere, Transform *transform, BoxCollider *other, Transform *otherTransform) {
          // Sphere and AABB
          Vec3f position = transform->position;
          Vec3f otherPosition = otherTransform->position;
           
          Vec3f lower = Vec3f(-other->width, -other->height, -other->depth);
          Vec3f upper = Vec3f(other->width, other->height, other->depth);
          
          Vec3f closest = (position - otherPosition).clamp(lower, upper) + otherPosition;
          float distance2 = closest.dst2(position);
          bool intersecting = distance2 <= (sphere->radius * sphere->radius);
          if (!intersecting || distance2 < 0.00001f) return ManifoldPoints();
          
          Vec3f AtoB = (closest - position).nor() * sphere->radius + position;
          Vec3f BtoA = closest;
                   
          Vec3f dir = BtoA - AtoB;
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
          
          Vec3f centerUnrotated = position - otherPosition;
          centerUnrotated = inverse.transform(centerUnrotated);
          Vec3f closestUnrotated = Vec3f(centerUnrotated).clamp(lower, upper) + otherPosition;
          centerUnrotated = centerUnrotated + otherPosition;
          
          float distance2 = closestUnrotated.dst2(centerUnrotated);
          bool intersecting = distance2 <= (sphere->radius * sphere->radius);
          if (!intersecting || distance2 < 0.00001f) return ManifoldPoints();
          
          Vec3f closestRotated = closestUnrotated - otherPosition;
          closestRotated = rotation.transform(closestRotated);
          closestRotated = closestRotated + otherPosition;
          
          
          Vec3f AtoB = (closestRotated - position).nor() * sphere->radius + position;
          Vec3f BtoA = closestRotated;
                   
          Vec3f dir = BtoA - AtoB;
          Vec3f normal = Vec3f(dir).nor();
          return ManifoldPoints(AtoB, BtoA, normal, dir.len());
     }
     ManifoldPoints GJK(Collider *collider, Transform *transform, Collider *other, Transform *otherTransform) {
          Simplex simplex;
          
          bool collided = GJK::contains_origin(simplex, collider, other, transform, otherTransform);
          if (!collided) return ManifoldPoints();
          
          return GJK::expanding_polytope(simplex, collider, other, transform, otherTransform);
     }
};

struct Object {
    Transform *transform;
    Collider *collider = 0;
    Vec3f collidingNormal = Vec3f(0.0f, 0.0f, 0.0f);
    ManifoldPoints lastManifoldPoints;
    Object *collidingObject = 0;
    std::function<void(Manifold, float)> onCollision;
    
    int index = 0;
    bool immovable = false;
    bool isTrigger = false;
    Mesh *mesh;
    
    
    Object() {
        transform = new Transform();
        mesh = new Mesh();
    }
    Object(bool immovable) : Object() {
        this->immovable = immovable;
    }
    virtual ~Object() {}
    
    void place(const Vec3f &to) {
        transform->position = to;
    }
    
    void place(float x, float y, float z) {
        transform->position = Vec3f(x, y, z);
    }
    void place_offset(float x, float y, float z) {
        transform->position = transform->position + Vec3f(x, y, z);
    }
    
    void rotate(float roll, float pitch, float yaw) {
        transform->rotation = transform->rotation.from_euler(roll, pitch, yaw);
    }
    void set_immovable(bool to) {
        immovable = to;
    }
    void set_trigger(bool to) {
        isTrigger = to;
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
    void set_collider(Collider *to) {
        this->collider = to;
    }
    
    void on_collision(const std::function<void(Manifold, float)> &collision) {
        this->onCollision = collision;
    }
    
    bool has_collided() { return (collidingObject); }
};

struct RigidBody : public Object {
    Vec3f velocity;
    Vec3f force;
    
    Vec3f angularVelocity;
    Vec3f torque;
    Vec3f inertiaParameters;
    float mass = 1.0f;
    float restitution = 0.5f;
    Tensor3x3 inertia;
    Tensor3x3 inverseInertia;
    
    RigidBody() : Object() {
          velocity = force = torque = Vec3f(0.0f, 0.0f, 0.0f);
          inertiaParameters = Vec3f(1.0f, 1.0f, 1.0f);
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
    void apply_instant_torque(const Vec3f &location, const Vec3f &direction) {
         Vec3f gradient = transform->position - location;
         Vec3f amount = gradient.crs(direction);
         
         torque = torque + amount;
    }
        
    void set_mass(float newMass) {
         mass = newMass;
    }
    void set_restitution(float newRestitution) {
         restitution = newRestitution;
    }
    void set_inertia_params(float x, float y, float z) {
         inertiaParameters.x = x;
         inertiaParameters.y = y;
         inertiaParameters.z = z;
    }
    void calculate_inertia_tensor(float x, float y, float z) {
         if (x == 0 || y == 0 || z == 0) {
              inertia.identity();
              return;
         }
         inertia.values[0][0] = mass * (y*y + z*z);
         inertia.values[1][1] = mass * (x*x + z*z);
         inertia.values[2][2] = mass * (x*x + y*y);
         
         inverseInertia = Tensor3x3(inertia).inverse();
    }
};
struct SphereObject : public RigidBody {
    float radius = 0.1f;
    SphereObject() : RigidBody() {
         this->collider = new SphereCollider(radius);
         
         this->set_structure(MeshGenerator::get_sphere_mesh(10, 10));
         this->mesh->scale(radius, radius, radius); 
    }
    SphereObject(float radius) : RigidBody() {
         this->radius = radius;
         this->collider = new SphereCollider(radius);
         
         
         this->set_structure(MeshGenerator::get_sphere_mesh(10, 10));
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
         this->collider = new ColliderGroup();
         
         for (auto &mesh : MeshGenerator::load_meshes_from_file(objFile)) {
              Vec3f centroid = MeshGenerator::generate_bounding_box_centroid(mesh);
              
              Transform *meshTransform = new Transform();
              meshTransform->position = Vec3f(centroid);
              BoxCollider *meshCollider = new BoxCollider(MeshGenerator::generate_bounding_box_sizes(mesh));
              
              ColliderGroup *group = (ColliderGroup*) this->collider; 
              group->append(meshCollider, meshTransform);
         }
         
         MeshGenerator::load_from_files(this->mesh, objFile, "");
    }
};
struct MeshObject : public RigidBody {
    MeshObject() : RigidBody() {}
    MeshObject(const std::string &objFile) : RigidBody() {
         this->collider = new MeshCollider(MeshGenerator::load_vertices(objFile));
         MeshGenerator::load_from_files(this->mesh, objFile, "");
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
     virtual void solve(const std::vector<Manifold> &manifolds) {};
};
struct PositionSolver : Solver {
     void solve(const std::vector<Manifold> &manifolds) override {
           for (auto &manifold : manifolds) {
                // Avoid updating if the objects have a very small collisison depth
                if (manifold.points.depth < 0.00001f) continue;
                
                Object *object1 = manifold.object1;
                Object *object2 = manifold.object2;
                
                int immovable1 = (int)object1->immovable;
                int immovable2 = (int)object2->immovable;
                
                Vec3f normal = manifold.points.normal;
                Vec3f displacement = normal * (1 / (float)std::max(1, immovable1 + immovable2));
                displacement = displacement * manifold.points.depth * 0.5f;
                
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
     void solve(const std::vector<Manifold> &manifolds) override {
           for (auto &manifold : manifolds) {
               // Avoid updating if the objects have a very small collisison depth 
               if (manifold.points.depth < 0.00001f) continue;
                
               RigidBody *object1 = dynamic_cast<RigidBody*>(manifold.object1);
               RigidBody *object2 = dynamic_cast<RigidBody*>(manifold.object2);
               if (object1 == nullptr || object2 == nullptr) continue;
               
               Vec3f position1 = object1->transform->position;
               Vec3f position2 = object2->transform->position;
               
               Vec3f normal = manifold.points.normal;
               Vec3f r1 = object1->transform->position - manifold.points.AtoB;
               Vec3f r2 = object2->transform->position - manifold.points.BtoA;
               
               Vec3f v1 = object1->velocity + r1.crs(object1->angularVelocity);
               Vec3f v2 = object2->velocity + r2.crs(object2->angularVelocity);
               Vec3f gradientVelocity = v1 - v2;
               
               Vec3f cp1 = (r1.crs(normal)).crs(r1);
               Vec3f cp2 = (r2.crs(normal)).crs(r2);
               Vec3f inertia1 = object1->inverseInertia * cp1;
               Vec3f inertia2 = object2->inverseInertia * cp2;
                       
               float dot = normal.dot(gradientVelocity);
               float dotInertia1 = normal.dot(inertia1);
               float dotInertia2 = normal.dot(inertia2);
               float restitution = object1->restitution + object2->restitution;
               float j = (1 + restitution) * dot / ((object1->mass + object2->mass) + dotInertia1 + dotInertia2);
                
               
               if (!object1->immovable) {
                   object1->velocity.x -= j * normal.x * object2->mass;
                   object1->velocity.y -= j * normal.y * object2->mass;
                   object1->velocity.z -= j * normal.z * object2->mass;
                   
                   Vec3f direction = normal * j;
                   object1->apply_instant_torque(manifold.points.AtoB, direction);
               }
               if (!object2->immovable) {
                   object2->velocity.x += j * normal.x * object1->mass;
                   object2->velocity.y += j * normal.y * object1->mass;
                   object2->velocity.z += j * normal.z * object1->mass;
                   
                   Vec3f direction = normal * -j;
                   object2->apply_instant_torque(manifold.points.BtoA, direction);
               }
          }
     }
};
struct ForceGenerator {
     virtual void apply(RigidBody *object, float timeTook) {};
};
struct GravityForce : ForceGenerator {
     Vec3f force;
     
     GravityForce(Vec3f force) : force(force) {}
     void apply(RigidBody *object, float timeTook) override {
          Vec3f acceleration = object->velocity * -1 + force;
          float mass = object->mass;
          
          object->force.x += acceleration.x * mass;
          object->force.y += acceleration.y * mass;
          object->force.z += acceleration.z * mass;
     }
};

struct WindForce : ForceGenerator {
     Vec3f direction;
     float strength;
     
     WindForce(Vec3f direction, float strength) : direction(direction), strength(strength) {
          this->direction.nor();
     }
     void apply(RigidBody *object, float timeTook) override {
          Vec3f force = direction * strength;
          
          object->force.x += force.x * timeTook;
          object->force.y += force.y * timeTook;
          object->force.z += force.z * timeTook;
     }
};


struct Constraint {
     virtual void apply() {};
     virtual void render() {};
};
struct SpringConstraint : Constraint {
     RigidBody *object1 = nullptr, *object2 = nullptr;
     
     float damping;
     float stiffness;
     float restLength;
     
     SpringConstraint(float restLength, float damping, float stiffness) : restLength(restLength), damping(damping), stiffness(stiffness) {}
     
     void apply() override {
          if (object1 == nullptr || object2 == nullptr) return;
          
          Vec3f position = object1->transform->position;
          Vec3f otherPosition = object2->transform->position;
          
          Vec3f velocity = object1->velocity;
          Vec3f otherVelocity = object2->velocity;
          
          Vec3f direction = otherPosition - position;
          float length2 = direction.len2();
          if (length2 > 0) {
              float length = sqrt(length2);
              float ax = (otherVelocity.x - velocity.x) * (otherPosition.x - position.x);
              float ay = (otherVelocity.y - velocity.y) * (otherPosition.y - position.y);
              float az = (otherVelocity.z - velocity.z) * (otherPosition.z - position.z);
              
              // Hooke's law
              float magnitude = (length - restLength) * stiffness;
              magnitude += (ax + ay + az) * damping / length;
              
              Vec3f spring = direction * (magnitude / length);
              
              object1->force.x += spring.x;
              object1->force.y += spring.y;
              object1->force.z += spring.z;
              
              object2->force.x -= spring.x;
              object2->force.y -= spring.y;
              object2->force.z -= spring.z;
         }
     }
     void render() override {
         if (object1 == nullptr || object2 == nullptr) return;
          
         Vec3f p1 = object1->transform->position;
         Vec3f p2 = object2->transform->position;
         
         BatchRenderer::draw_spring(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
     }
     
     void link(RigidBody *object1, RigidBody *object2) {
          this->object1 = object1;
          this->object2 = object2;
     }
};

class ObjectLevel {
    public:
        virtual void load() {}
        virtual void update(float timeTook) {}
        
        void add_object(Object *body) {
             body->index = lastIndex;
             if (body->collider) {
                  body->collider->boundingRadius = body->collider->compute_bounding();
             }
             
             objects.push_back(body);
             lastIndex++;
        }
        void remove_object(Object *body) {
             objects.erase(objects.begin() + body->index);
        }
        
        void add_solver(Solver *solver) {
             solvers.push_back(solver);
        }
        
        void send_collision_callbacks(const std::vector<Manifold> &collisions, float timeTook) {
             for (auto &manifold : collisions) {
                  auto &callback1 = manifold.object1->onCollision;
                  auto &callback2 = manifold.object2->onCollision;
                  
                  if (callback1) callback1(manifold, timeTook);
                  if (callback2) callback2(manifold, timeTook);
             }
        }
        
        
        void resolve_collisions(float timeTook) {
             manifolds.clear();
             triggers.clear();
             
             // Collision detection
             for (auto &object1 : objects) {
                  float temporaryDot = 0.0f;
                  Vec3f normal;
                  for (auto &object2 : objects) {
                       if (object1->index == object2->index) break;
                       if (!object1->collider || !object2->collider) continue;
                       if (object1->immovable && object2->immovable) continue;
                       
                       float radius1 = object1->collider->boundingRadius;
                       float radius2 = object2->collider->boundingRadius;
                       float distance2 = object1->transform->position.dst2(object2->transform->position);
                       if (distance2 > (radius1 + radius2) * (radius1 + radius2)) continue;
                       
                       ManifoldPoints points = object1->collider->test(object1->transform, object2->collider, object2->transform);
                       if (!points.collided) continue;
                            
                       Manifold manifold = Manifold(object1, object2, points);
                       bool trigger = object1->isTrigger || object2->isTrigger;
                           
                       float dotProductGravity = (points.normal * -1).dot(Vec3f(0.0f, -1.0f, 0.0f));
                       if (dotProductGravity > temporaryDot) {
                           normal = points.normal;
                           temporaryDot = dotProductGravity;
                       }
                       
                       object1->collidingNormal = normal;
                       object2->collidingNormal = normal;
                       object1->lastManifoldPoints = points;
                       object2->lastManifoldPoints = points;  
                       if (!object2->isTrigger) object1->collidingObject = object2;
                       if (!object1->isTrigger) object2->collidingObject = object1;
                            
                       if (trigger) triggers.emplace_back(manifold);
                       else manifolds.emplace_back(manifold);
                  }
             }
             
             for (auto &object1 : objects) {
                  float temporaryDot = 10000.0f;
                  Vec3f normal;
                  
                  for (auto &object2 : objects) {
                       if (object1->index == object2->index) continue;
                       if (!object1->collider || !object2->collider) continue;
                       if (object1->immovable && object2->immovable) continue;
                       
                       float radius1 = object1->collider->boundingRadius;
                       float radius2 = object2->collider->boundingRadius;
                       float distance2 = object1->transform->position.dst2(object2->transform->position);
                       if (distance2 > (radius1 + radius2) * (radius1 + radius2)) continue;
                       
                       ColliderGroup *group = dynamic_cast<ColliderGroup*>(object2->collider);
                       
                       if (group == nullptr) continue;
                       
                       for (auto &pair : group->colliders) {
                            Collider *collider = pair.first;
                            Transform *transform = pair.second;
                            
                            Vec3f position = Vec3f(transform->position);
                            position = object2->transform->rotation.transform(position);
                            position = position + object2->transform->position;
                            
                            Transform *newTransform = new Transform();
                            newTransform->position = position;
                            newTransform->rotation = Quaternion(transform->rotation).multiply_left(object2->transform->rotation);
                            
                            ManifoldPoints points = object1->collider->test(object1->transform, collider, newTransform);
                            if (!points.collided) continue;
                            
                            Manifold manifold = Manifold(object1, object2, points);
                            bool trigger = object1->isTrigger || object2->isTrigger;
                            
                            float dotProductGravity = (points.normal * -1).dot(Vec3f(0.0f, -1.0f, 0.0f));
                            if (dotProductGravity < temporaryDot) {
                                 normal = points.normal * -1;
                                 temporaryDot = dotProductGravity;
                            }
                            
                            
                            object1->collidingNormal = normal;
                            object2->collidingNormal = normal;
                            object1->lastManifoldPoints = points;
                            object2->lastManifoldPoints = points;    
                            if (!object2->isTrigger) object1->collidingObject = object2;
                            if (!object1->isTrigger) object2->collidingObject = object1;
                            
                            
                            if (trigger) triggers.emplace_back(manifold);
                            else manifolds.emplace_back(manifold); 
                       }
                  }
             }
             
             // Don't resolve collisions with triggers
             for (auto &solver : solvers) {
                  solver->solve(manifolds);
             }
             send_collision_callbacks(manifolds, timeTook);
             send_collision_callbacks(triggers, timeTook);
        }
        
        std::vector<Object*> &get_objects() { return objects; }
        
   protected:
        std::vector<Object*> objects;
        std::vector<Solver*> solvers;
        std::vector<Manifold> manifolds, triggers;
        
        int lastIndex = 0;
};

class PhysicsLevel : public ObjectLevel {
    public:
        Vec3f gravity;
        
        void load() override {
             simulationSteps = 5;
             gravity = Vec3f(0.0f, -9.81f, 0.0f);
             
             objects.clear();
             solvers.clear();
             forces.clear();
             constraints.clear();
             
             add_solver(new PositionSolver());
             add_solver(new ElasticImpulseSolver());
             
             add_force(new GravityForce(gravity));
        }
        
        void update(float timeTook) override {
             float stepSize = timeTook / (float)simulationSteps;
             for (int i = 0; i < simulationSteps; i++) {
                  update_with_sub_steps(stepSize);
             }
        }
        
        void update_with_sub_steps(float timeTook) {
             resolve_collisions(timeTook);
             
             // Reseting the forces + applying
             for (auto &object : objects) {
                  RigidBody *body = dynamic_cast<RigidBody*>(object);
                  if (body == nullptr) continue;
                  
                  Vec3f params = body->inertiaParameters;
                  body->calculate_inertia_tensor(params.x, params.y, params.z);
                  
                  body->force = Vec3f(0.0f, 0.0f, 0.0f);
                  if (body->immovable) {
                       body->velocity = Vec3f(0.0f, 0.0f, 0.0f);
                       body->angularVelocity = Vec3f(0.0f, 0.0f, 0.0f);
                  }
                  
                  for (auto &force : forces) {
                       force->apply(body, timeTook);
                  }
             }
             for (auto &constraint : constraints) {
                  constraint->apply();
             }
             
             for (auto &object : objects) {
                  if (object->immovable) continue;
                  
                  RigidBody *body = dynamic_cast<RigidBody*>(object);
                  if (body == nullptr) continue;
                  
                  // Linear motion
                  float nextForceX = body->force.x + (body->force.x * timeTook);
                  float nextForceY = body->force.y + (body->force.y * timeTook);
                  float nextForceZ = body->force.z + (body->force.z * timeTook);
                  
                  float correctForceX = (body->force.x + nextForceX) / 2.0f * timeTook;
                  float correctForceY = (body->force.y + nextForceY) / 2.0f * timeTook;
                  float correctForceZ = (body->force.z + nextForceZ) / 2.0f * timeTook;
                  
                  // Add velocity with acceleration
                  float nextVelocityX = body->velocity.x + correctForceX / body->mass;
                  float nextVelocityY = body->velocity.y + correctForceY / body->mass;
                  float nextVelocityZ = body->velocity.z + correctForceZ / body->mass;
                  
                  float correctVelocityX = (body->velocity.x + nextVelocityX) / 2.0f * timeTook;
                  float correctVelocityY = (body->velocity.y + nextVelocityY) / 2.0f * timeTook;
                  float correctVelocityZ = (body->velocity.z + nextVelocityZ) / 2.0f * timeTook;
                  
                  // Rotational motion
                  Vec3f angularAcceleration = body->inverseInertia.multiply(body->torque);
                 
                  float nextAAccelerationX = angularAcceleration.x + (angularAcceleration.x * timeTook);
                  float nextAAccelerationY = angularAcceleration.y + (angularAcceleration.y * timeTook);
                  float nextAAccelerationZ = angularAcceleration.z + (angularAcceleration.z * timeTook);
                  
                  float correctAAccelerationX = (angularAcceleration.x + nextAAccelerationX) / 2.0f * timeTook;
                  float correctAAccelerationY = (angularAcceleration.y + nextAAccelerationY) / 2.0f * timeTook;
                  float correctAAccelerationZ = (angularAcceleration.z + nextAAccelerationZ) / 2.0f * timeTook;
                  
                  float nextAVelocityX = body->angularVelocity.x + correctAAccelerationX;
                  float nextAVelocityY = body->angularVelocity.y + correctAAccelerationY;
                  float nextAVelocityZ = body->angularVelocity.z + correctAAccelerationZ;
                  
                  float correctAVelocityX = (body->angularVelocity.x + nextAVelocityX) / 2.0f * timeTook;
                  float correctAVelocityY = (body->angularVelocity.y + nextAVelocityY) / 2.0f * timeTook;
                  float correctAVelocityZ = (body->angularVelocity.z + nextAVelocityZ) / 2.0f * timeTook;
                  
                  
                  
                  body->velocity.x += correctForceX / body->mass;
                  body->velocity.y += correctForceY / body->mass;
                  body->velocity.z += correctForceZ / body->mass;
                  
                  
                  body->transform->position.x += correctVelocityX;
                  body->transform->position.y += correctVelocityY;
                  body->transform->position.z += correctVelocityZ;
                  
                  body->angularVelocity.x += correctAAccelerationX;
                  body->angularVelocity.y += correctAAccelerationY;
                  body->angularVelocity.z += correctAAccelerationZ;
                  
                  body->transform->rotation = body->transform->rotation.multiply_left(Quaternion().from_euler(correctAVelocityX, correctAVelocityY, correctAVelocityZ));
             }
             for (auto &object : objects) {
                  RigidBody *body = dynamic_cast<RigidBody*>(object);
                  if (body == nullptr) continue;
                  body->torque = Vec3f(0.0f, 0.0f, 0.0f);
             }
        }
        
        void add_force(ForceGenerator *generator) {
             forces.push_back(generator);
        }
        void add_constraint(Constraint *constraint) {
             constraints.push_back(constraint);
        }
        std::vector<Constraint*> &get_constraints() { return constraints; }
        
    private:
        int simulationSteps;
        std::vector<ForceGenerator*> forces;
        std::vector<Constraint*> constraints;
};

using Ball_t = SphereObject;
class Level {
    public:
        bool completedLevel;
        
        static Level &get()
        {
            static Level ins;
            return ins;
        }
        void load() {
            std::vector<std::string> textures = {
                 "resources/textures/side.png",
                 "resources/textures/side.png",
                 "resources/textures/top.png",
                 "resources/textures/bottom.png",
                 "resources/textures/side.png",
                 "resources/textures/side.png"
            };
            
            objectShader = new Shader("resources/shaders/object.vert", "resources/shaders/object.frag");
            skyboxShader = new Shader("resources/shaders/skybox.vert", "resources/shaders/skybox.frag");
            skybox = new Skybox(textures);
            
            level.load();
                         
            ball = new SphereObject(0.35f);
            ball->set_material(Materials::aluminium);
            ball->set_restitution(0.2f);
            
            this->start(0);
        }
        void start(int levelIndex) {
            level.get_objects().clear();
            level.get_constraints().clear();
            
            ball->velocity = Vec3f(0.0f, 0.0f, 0.0f);
            level.add_object(ball);
           
            if (levelIndex == 0) {
                       ball->place(2.0f, 10.0f, 0.0f);
                       
                       add_box(Vec3f(0.0f, -1.0f, 0.0f), 10.0f, 0.5f, 10.0f, "grass.png", Materials::grass);
                       add_box(Vec3f(-17.0f, -1.0f, 0.0f), 4.0f, 0.5f, 0.5f, "grass.png", Materials::grass);
                       add_box(Vec3f(-26.0f, -1.0f, 0.0f), 1.0f, 0.5f, 1.0f, "grass.png", Materials::grass);
                       add_box(Vec3f(-33.0f, -1.0f, 7.0f), Vec3f(0.0f, 1.0f, 0.0f), 5.0f, 0.5f, 0.5f, "grass.png", Materials::grass);
                       add_box(Vec3f(-43.0f, -1.0f, 17.0f), 5.0f, 0.5f, 5.0f, "grass.png", Materials::grass);
                       add_box(Vec3f(-43.0f, 1.5f, 17.0f), 2.0f, 2.0f, 2.0f, "wood.png", Materials::wood);
                       add_box(Vec3f(-46.0f, 0.5f, 17.0f), 1.0f, 1.0f, 1.0f, "wood.png", Materials::wood);
                      
                      
                       add_finish(Vec3f(-43.0f, 5.5f, 17.0f), 1.5f, 2.0f, 1.5f);
                      
                       add_model(Vec3f(0.0f, 0.0f, 0.0f), "table.obj", "wood.png", Materials::wood);
                       add_model(Vec3f(2.0f, 0.0f, 0.0f), "chair.obj", "wood.png", Materials::wood); 
                       ModelObject *model = add_model(Vec3f(-2.0f, 0.0f, 0.0f), "chair.obj", "wood.png", Materials::wood);                      
                       model->rotate(0.0f, M_PI, 0.0f);
                          
            } else if (levelIndex == 1) {
                       ball->place(0.0f, 5.0f, 0.0f);
                       
                       add_box(Vec3f(0.0f, -1.0f, 0.0f), 5.0f, 0.5f, 5.0f, "grass.png", Materials::grass);
                       add_box(Vec3f(6.0f, 0.5f, 0.0f), 1.0f, 2.0f, 5.0f, "grass.png", Materials::grass);
                       add_box(Vec3f(4.0f, 0.5f, 0.0f), 1.0f, 1.0f, 1.0f, "wood.png", Materials::wood);
                       
                       add_box(Vec3f(10.0f, 2.0f, 0.0f), 3.0f, 0.5f, 5.0f, "grass.png", Materials::grass);
                       add_box(Vec3f(15.0f, -4.0f, 0.0f), 1.0f, 0.5f, 7.0f, "grass.png", Materials::grass);
                       add_box(Vec3f(15.0f, -2.5f, 6.0f), 1.0f, 1.0f, 1.0f, "wood.png", Materials::wood);
                       add_box(Vec3f(25.0f, -1.0f, 6.0f), 5.0f, 0.5f, 0.25f, "grass.png", Materials::grass);
                       add_box(Vec3f(19.0f, -1.0f, 6.0f), 1.0f, 0.5f, 1.0f, "grass.png", Materials::grass);
                       
                       // Floor
                       add_box(Vec3f(37.0f, -7.0f, 6.0f), 3.0f, 0.5f, 3.0f, "wood.png", Materials::wood);
                       
                       // To other platform
                       add_box(Vec3f(45.0f, -2.0f, 6.0f), 3.0f, 0.5f, 1.0f, "grass.png", Materials::grass);
                       add_finish(Vec3f(50.0f, 1.0f, 6.0f), 1.5f, 2.0f, 1.5f);
                       
                       // Stairs
                       add_box(Vec3f(34.5f, -5.5f, 8.5f), 0.5f, 1.0f, 0.5f, "wood.png", Materials::wood);
                       add_box(Vec3f(34.5f, -4.5f, 7.5f), 0.5f, 2.0f, 0.5f, "wood.png", Materials::wood);
                       add_box(Vec3f(34.5f, -3.5f, 6.5f), 0.5f, 3.0f, 0.5f, "wood.png", Materials::wood);
                       
                       // Walls
                       add_box(Vec3f(33.5f, -3.5f, 6.5f), 0.5f, 4.0f, 4.0f, "wood.png", Materials::wood);
                       
                       // Ceilling
                       BoxObject *box = add_box(Vec3f(37.0f, 4.0f, 6.0f), 3.0f, 0.5f, 3.0f, "wood.png", Materials::wood);
                       ModelObject *bowl = add_model(Vec3f(37.0f, 2.0f, 6.0f), "bowl.obj", "wood.png", Materials::wood);
                       
                       MeshObject *mesh = add_mesh_object(Vec3f(-2.0f, 1.5f, 0.0f), "icosahedron_TESS.obj", "", Materials::copper);
                       mesh->set_immovable(false);
                       
                       bowl->set_mass(1.0f);
                       bowl->set_immovable(false);
                       
                       add_sphere(Vec3f(37.0f, 3.0f, 6.0f), 0.35f, Materials::copper);
                       
                       SpringConstraint *constraint = new SpringConstraint(3.0f, 7.0f, 10.0f);
                       constraint->link(box, bowl);
                       add_constraint(constraint);
            } else {
                       ball->place(0.0f, 1.0f, 0.0f);
                       
                       add_box(Vec3f(0.0f, -1.0f, 0.0f), 10.0f, 0.5f, 10.0f, "grass.png", Materials::grass);
            }
            
            
            completedLevel = false;
        }
        
        void update(float timeTook) {
            level.update(timeTook);
        }
        
        void render(Camera *camera) {
            bool debug = false;
            if (!debug) {
                 // Skybox
                 skyboxShader->use();
                 skyboxShader->set_uniform_mat4("view", camera->get_view());
                 skyboxShader->set_uniform_mat4("projection", camera->get_projection());
            
                 skybox->render(skyboxShader);
              
                 // Objects
                 objectShader->use();
                 objectShader->set_uniform_mat4("view", camera->get_view());
                 objectShader->set_uniform_mat4("projection", camera->get_projection());
                 objectShader->set_uniform_vec3f("viewPosition", camera->position.x, camera->position.y, camera->position.z);
        
         
                 for (auto &object : level.get_objects()) {
                       object->mesh->rotate(object->transform->rotation);
                       object->mesh->translate(object->transform->position);
                 
                       object->mesh->render(objectShader);
                 }
            } else {
                 for (auto &object : level.get_objects()) {
                       BoxCollider *box = dynamic_cast<BoxCollider*>(object->collider);
                       SphereCollider *sphere = dynamic_cast<SphereCollider*>(object->collider);
                       ColliderGroup *group = dynamic_cast<ColliderGroup*>(object->collider);
                       
                       Vec3f highlight = object->collidingObject ? Vec3f(0.0f, 1.0f, 0.0f) : Vec3f(1.0f, 1.0f, 1.0f);
                       Vec3f p = object->transform->position;
                       
                       if (box != nullptr) {
                            BatchRenderer::draw_box_outline(p.x, p.y, p.z, box->width, box->height, box->depth, object->transform->rotation, highlight); 
                       }
                       if (sphere != nullptr) {
                            BatchRenderer::draw_sphere_outline(p.x, p.y, p.z, sphere->radius, highlight);
                       }
                       if (group != nullptr) {
                            for (auto &pair : group->colliders) {
                                 BoxCollider *collider = (BoxCollider*) pair.first;
                                 Vec3f pos = pair.second->position + object->transform->position;
                                 BatchRenderer::draw_box_outline(pos.x, pos.y, pos.z, collider->width, collider->height, collider->depth, object->transform->rotation, highlight); 
                       
                            }
                       }
                 }
                 UIRenderer::draw_string("Collider debug", -0.9f, 0.87f, 0.7f, 0.7f);
            }
            
            for (auto &constraint : level.get_constraints()) {
                 constraint->render();
            }
            BatchRenderer::render(camera);
        }
        void dispose() {
            for (auto &object : level.get_objects()) {
                 object->mesh->dispose();
            }
            skybox->dispose();
            
            objectShader->clear();
            skyboxShader->clear();
        }
        /*
        void load_level(const std::string &fileName) {
            std::ifstream read(fileName);
            if (!read.is_open() || read.fail()) {
                printf("Couldn't open the level file.\n");
                return;
            }   
       
            std::string line; 
            while (std::getline(read, line)) {
                  if (!line.compare("") || !line.compare(" ")) {
                       continue;
                  }
              
                  std::istringstream stream(line);
                  std::string key;
                  stream >> key;
              
                  if (!key.compare("playerPosition")) {
                       float x, y, z;
                       stream >> x >> y >> z;
                       
                       ball->place(x, y, z);
                  }
                  
                  else if (!key.compare("box")) {
                       Vec3f position, rotation;
                       float width, height, depth;
                       std::string textureLocation, materialLocation;
                       
                       stream >> position.x >> position.y >> position.z
                              >> width >> height >> depth
                              >> rotation.x >> rotation.y >> rotation.z
                              >> textureLocation >> materialLocation;
                              
                       //add_box(position, rotation, width, height, depth, textureLocation, materials[materialLocation]);
                  }
                  else if (!key.compare("sphere")) {
                       Vec3f position;
                       float radius;
                       std::string materialLocation;
                       
                       stream >> position.x >> position.y >> position.z
                              >> radius
                              >> materialLocation;
                       
                       //add_sphere(position, radius, materials[materialLocation]);
                  }
                  else if (!key.compare("model")) {
                       Vec3f position;
                       std::string modelFile;
                       
                       stream >> position.x >> position.y >> position.z
                              >> modelFile;
                       
                       //add_model(position, modelFile);
                  }
                  else if (!key.compare("finish")) {
                       Vec3f position, rotation;
                       float width, height, depth;
                       std::string levelName;
                       
                       stream >> position.x >> position.y >> position.z
                              >> width >> height >> depth
                              >> levelName;
                       
                       //add_finish(position, width, height, depth, levelName);
                  }
                  
                  else if (!key.compare("setMass")) {
                       int index;
                       float amount;
                       
                       stream >> index >> amount;
                       
                       ((RigidBody*) level.get_objects().at(index))->set_mass(amount);
                  }
                  else if (!key.compare("isImmovable")) {
                       int index;
                       bool immovable;
                       
                       stream >> index >> immovable;
                       
                       ((RigidBody*) level.get_objects().at(index))->set_immovable(immovable);
                  }
                  
                  else if (!key.compare("spring")) {
                       int index1, index2;
                       float restLength, damping, stiffness;
                       
                       stream >> index1 >> index2
                              >> restLength >> damping >> stiffness;
                       
                       
                       RigidBody *object1 = (RigidBody*) level.get_objects().at(index1);
                       RigidBody *object2 = (RigidBody*) level.get_objects().at(index2);
                       
                       SpringConstraint *constraint = new SpringConstraint(restLength, damping, stiffness);
                       constraint->link(object1, object2);
                       
                       add_constraint(constraint);
                  }
             }
        }
        */
        BoxObject *add_box(const Vec3f &position, const Vec3f &rotation, float width, float height, float depth, const std::string &textureFile = "", const Material &material = Materials::defaultMaterial) {
            BoxObject *box = new BoxObject(width, height, depth);
            
            box->set_immovable(true);
            box->place(position.x, position.y, position.z);
            box->rotate(rotation.x, rotation.y, rotation.z);
            if (textureFile.compare("")) box->set_texture("resources/textures/" + textureFile);
            box->set_material(material);
            
            level.add_object(box);
            
            return box;
        }
        BoxObject *add_box(const Vec3f &position, float width, float height, float depth, const std::string &textureFile = "", const Material &material = Materials::defaultMaterial) {
            return add_box(position, Vec3f(0.0f, 0.0f, 0.0f), width, height, depth, textureFile, material);
        }
        SphereObject *add_sphere(const Vec3f &position, float radius, const Material &material = Materials::defaultMaterial) {
            SphereObject *sphere = new SphereObject(radius);
            sphere->place(position.x, position.y, position.z);
            sphere->set_material(material);
            
            level.add_object(sphere);
            return sphere;       
        }
        ModelObject *add_model(const Vec3f &position, const std::string &modelFile, const std::string &textureFile, const Material &material = Materials::defaultMaterial) {
            ModelObject *model = new ModelObject("resources/objects/" + modelFile);
            
            model->set_immovable(true);
            model->place(position.x, position.y, position.z);
            model->set_material(material);
            if (textureFile.compare("")) model->set_texture("resources/textures/" + textureFile);
            
            level.add_object(model);
            return model;
        }
        MeshObject *add_mesh_object(const Vec3f &position, const std::string &modelFile, const std::string &textureFile, const Material &material = Materials::defaultMaterial) {
            MeshObject *obj = new MeshObject("resources/objects/" + modelFile);
            
            obj->set_immovable(true);
            obj->place(position.x, position.y, position.z);
            obj->set_material(material);
            if (textureFile.compare("")) obj->set_texture("resources/textures/" + textureFile);
            
            level.add_object(obj);
            return obj;
        }
        Object *add_finish(const Vec3f &position, float width, float height, float depth) {
            Object *trigger = new Object(true);
            trigger->place(position.x, position.y, position.z);
            trigger->set_collider(new BoxCollider(width, height, depth));
            trigger->on_collision([&](Manifold collision, float timeTook) {
                  if (collision.object2 == ball) {
                      completedLevel = true;
                  }
            });
            trigger->set_trigger(true);
            level.add_object(trigger);
            
            return trigger;
        }
        
        void add_force(ForceGenerator *generator) {
            level.add_force(generator);
        }
        void add_constraint(Constraint *constraint) {
            level.add_constraint(constraint);
        }
        
        Ball_t *get_ball() { return ball; }
         
    private:
        Level() {}
        ~Level() {}
    public:
        Level(Level const&) = delete;
        void operator = (Level const&) = delete; 
        
    protected:
        Shader *objectShader;
        Shader *skyboxShader;
        Skybox *skybox;
        
        PhysicsLevel level;
        Ball_t *ball;
};

class BallControls {
    public:
        bool orbits = false;
        BallControls(Camera *camera, Ball_t *ball) {
            this->camera = camera;
            this->ball = ball;
        }
        void handle_event(SDL_Event ev, float timeTook) {
            int w = 0, h = 0;
            SDL_GetWindowSize(windows, &w, &h);
            
            int clickX, clickY;
            SDL_GetMouseState(&clickX, &clickY);
            
            if (ev.type == SDL_MOUSEMOTION) {       
                float sensitivity = 0.1f;
                if (orbits) {
                    if (clickY < h / 2) {
                        camera->rotationX -= ev.motion.xrel * sensitivity * timeTook;
                        camera->rotationY += ev.motion.yrel * sensitivity * timeTook;
                    }
                } else {
                    if (clickY < h / 2) {
                        camera->rotationX -= ev.motion.xrel * sensitivity * timeTook;
                        camera->rotationY -= ev.motion.yrel * sensitivity * timeTook;
                    }
                }
                
                if (camera->rotationX > 2 * M_PI) camera->rotationX = 0;
                if (camera->rotationX < 0) camera->rotationX = 2 * M_PI;
               
                if (camera->rotationY > (89.0f / 180.0f * M_PI)) camera->rotationY = (89.0f / 180.0f * M_PI);
                if (camera->rotationY < -(89.0f / 180.0f * M_PI)) camera->rotationY = -(89.0f / 180.0f * M_PI);
              
                float speed = 7.0f;
                Vec3f vel = Vec3f(cos(camera->rotationX),
                                  0.0f,
                                  sin(camera->rotationX));
                Vec3f v = vel * speed * timeTook;
                if (orbits) v = v * -1;
                
                if (clickY > h / 2 && clickX < w / 2) {
                    ball->velocity = ball->velocity + v;
                }
                else if (clickY > h / 2 && clickX > w / 2) {
                    ball->velocity = ball->velocity - v;
                }
            }
            if (ev.type == SDL_MOUSEBUTTONUP) {
                if (clickY > h / 1.3f) {
                     jump(Vec3f(0.0f, -1.0f, 0.0f));
                }
            }
        }
        void jump(const Vec3f &gravity) {
            if (!ball->has_collided()) return;
            if (ball->collidingNormal.len2() < 0.1f) return;
            
            float strength = 7.5f;
            Vec3f normal = ball->collidingNormal;
            
            if (normal.dot(gravity) < 0) {
                 Vec3f offset = normal * 0.1f;
                 Vec3f velocity = normal * strength;
                 
                 ball->transform->position = ball->transform->position + offset;
                 ball->velocity = ball->velocity + (velocity * (1 / ball->mass));
                 
                 
                 RigidBody *body = dynamic_cast<RigidBody*>(ball->collidingObject);
                 if (body != nullptr) {
                      if (!body->immovable) {
                           Vec3f impulse = velocity * (1 / body->mass);
                           body->velocity = body->velocity - impulse;
                           body->apply_instant_torque(ball->lastManifoldPoints.AtoB, impulse);
                      }
                      body->collidingObject = 0;
                 }
                 
                 ball->collidingObject = 0;
            }
        }
        void update() {
            if (orbits) {
                 float distance = 5.0f;
                 camera->lookingAt = ball->transform->position;
                 camera->position = ball->transform->position + Vec3f(cos(camera->rotationX) * cos(camera->rotationY) * distance, sin(camera->rotationY) * distance, sin(camera->rotationX) * cos(camera->rotationY) * distance);
            } else {
                 camera->position = ball->transform->position;
            }
        }
    private:
        Ball_t *ball;
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

class Aluminium3D : public Game {
    Camera *camera;
    BallControls *controls;
    float completedTimer = 0.0f;
    int levelIndex = 0;
    float timer = 0.0f;
    
    int fps = 0;
    float framesPerSecond = 0.0f;
    float lastTime = 0.0f;
    bool orbitMode = true;
           
    public:  
       void init() override {
           displayName = "Aluminium 3D";
       }
       void load() override {
           Materials::load();
           FreeType::get().load();
           UIRenderer::load();
           BatchRenderer::load();
           Level::get().load();
           
           
           camera = new Camera();
           camera->useRotation = !orbitMode;
           controls = new BallControls(camera, Level::get().get_ball());
           controls->orbits = orbitMode;
       }
       void handle_event(SDL_Event ev, float timeTook) override {
           controls->handle_event(ev, timeTook);
       }
       void update(float timeTook) override {
           if (!Level::get().completedLevel) {
                timer += timeTook / SDL_GetPerformanceFrequency() * 1000000;
           }
           
           Level::get().update(timeTook);
           
           controls->update();
           camera->update();
           
           if (Level::get().completedLevel) {
                completedTimer += timeTook;
                
                float duration = 1.25f;
                float clamp = std::max(0.0f, std::min(completedTimer / duration, 1.0f));
                float alpha = (3.0 - clamp * 2.0) * clamp * clamp;
                
                Vec2f start = Vec2f(0.0f, 1.25f);
                Vec2f end = Vec2f(0.0f, 0.25f);
                Vec2f position = start.interpolate(end, alpha);
                UIRenderer::draw_string_centered("Level complete!", position.x, position.y);
                
                if (completedTimer >= 5.0f) {
                     levelIndex++;
                     Level::get().start(levelIndex);
                     controls = new BallControls(camera, Level::get().get_ball());
                     controls->orbits = orbitMode;
                     
                     completedTimer = 0.0f;
                     timer = 0.0f;
                }
           }
           Vec3f ballPosition = Level::get().get_ball()->transform->position;
           UIRenderer::draw_string("FPS: " + std::to_string(fps), -0.9f, 0.8f, 0.5f, 0.5f);
           UIRenderer::draw_string("Timer: " + std::to_string((int)timer) + " seconds", -0.9f, 0.7f, 0.5f, 0.5f);
           UIRenderer::draw_string("Position: (" + std::to_string((int)ballPosition.x) + ", " + std::to_string((int)ballPosition.y) + ", " + std::to_string((int)ballPosition.z) + ")", 0.5f, 0.8f, 0.5f, 0.5f);
          
           Level::get().render(camera);
           UIRenderer::render();
           
           calculate_FPS();
       }
       
       void dispose() override {  
           Level::get().dispose();
           UIRenderer::dispose();
           BatchRenderer::dispose();
           FreeType::get().dispose();
       }
       
       void calculate_FPS() {
           float currentTime = SDL_GetTicks() * 0.001f;
           framesPerSecond++;
           if (currentTime - lastTime > 1.0f) {
                lastTime = currentTime;
                fps = (int) framesPerSecond;
                framesPerSecond = 0.0f;
           }
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
    
    
    Aluminium3D game;
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
    glEnable(GL_BLEND);
    glEnable(GL_POLYGON_OFFSET_FILL);
    
    glDepthFunc(GL_LESS);
    glCullFace(GL_FRONT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPolygonOffset(1, 0);
    
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