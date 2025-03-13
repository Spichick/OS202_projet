#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <SDL2/SDL.h>
#include <vector>
#include <memory>
#include <cstdint>

class Displayer
{
public:
    // 修改构造函数声明，添加 scale_factor
    Displayer( std::uint32_t width, std::uint32_t height, int scale_factor );

    // 禁止拷贝和移动
    Displayer( Displayer const & ) = delete;
    Displayer( Displayer      && ) = delete;

    // 析构函数
    ~Displayer();

    // 更新窗口
    void update( std::vector<std::uint8_t> const & vegetation_global_map,
                 std::vector<std::uint8_t> const & fire_global_map );

    // 修改 init_instance()，添加 scale_factor
    static std::shared_ptr<Displayer> init_instance( std::uint32_t t_width, std::uint32_t t_height, int scale_factor );

    static std::shared_ptr<Displayer> instance();

private:
    static std::shared_ptr<Displayer> unique_instance; // 单例指针

    SDL_Window* m_pt_window{nullptr};       // SDL 窗口
    SDL_Renderer* m_pt_renderer{nullptr};   // SDL 渲染器
    SDL_Surface* m_pt_surface{nullptr};     // SDL 窗口表面

    int m_scale_factor;  // 新增成员变量，存储缩放因子
};

#endif // DISPLAY_HPP
