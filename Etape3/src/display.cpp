#include <cassert>
#include <stdexcept>
#include <string> 
#include "display.hpp"

using namespace std::string_literals;

std::shared_ptr<Displayer> Displayer::unique_instance{nullptr};

Displayer::Displayer(std::uint32_t t_width, std::uint32_t t_height, int scale_factor)
    : m_scale_factor(scale_factor)
{
    // Initialisation du contexte pour SDL :
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        std::string err_msg = "Erreur lors de l'initialisation de SDL : "s + std::string(SDL_GetError());
        throw std::runtime_error(err_msg);
    }

    // Ajuster la taille de la fenêtre en fonction du facteur d'échelle
    t_width *= scale_factor;
    t_height *= scale_factor;

    // Création de la fenêtre et du moteur de rendu
    SDL_CreateWindowAndRenderer(t_width, t_height, 0, &m_pt_window, &m_pt_renderer);
    if (m_pt_window == nullptr)
    {
        std::string err_msg = "Erreur lors de la création de la fenêtre : "s + std::string(SDL_GetError());
        throw std::runtime_error(err_msg);
    }
    if (m_pt_renderer == nullptr)
    {
        std::string err_msg = "Erreur lors de la création du moteur de rendu : "s + std::string(SDL_GetError());
        throw std::runtime_error(err_msg);
    }

    // Récupération de la surface
    m_pt_surface = SDL_GetWindowSurface(m_pt_window);
    if (m_pt_surface == nullptr)
    {
        std::string err_msg = "Erreur lors de la récupération de la surface : "s + std::string(SDL_GetError());
        throw std::runtime_error(err_msg);
    }
}
// ====================================================================================================================
Displayer::~Displayer()
{
    SDL_DestroyRenderer(m_pt_renderer);
    SDL_DestroyWindow(m_pt_window);
    SDL_Quit();
}
// ====================================================================================================================
void
Displayer::update(std::vector<std::uint8_t> const& vegetation_global_map,
                  std::vector<std::uint8_t> const& fire_global_map)
{
    int w, h;
    SDL_GetWindowSize(m_pt_window, &w, &h);

    // 计算缩放后的尺寸
    w /= m_scale_factor;
    h /= m_scale_factor;

    SDL_SetRenderDrawColor(m_pt_renderer, 0, 0, 0, 255);
    SDL_RenderClear(m_pt_renderer);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            SDL_SetRenderDrawColor(m_pt_renderer, fire_global_map[j + w * i], vegetation_global_map[j + w * i], 0, 255);
            SDL_Rect pixel_rect = { j * m_scale_factor, (h - i - 1) * m_scale_factor, m_scale_factor, m_scale_factor };
            SDL_RenderFillRect(m_pt_renderer, &pixel_rect);
        }
    }

    SDL_RenderPresent(m_pt_renderer);
}
// ####################################################################################################################
//                      Définition des méthodes statiques associées au pattern singleton utilisé
std::shared_ptr<Displayer> 
Displayer::init_instance(std::uint32_t t_width, std::uint32_t t_height, int scale_factor)
{
    assert(("L'initialisation de l'instance ne doit etre appele qu'une seule fois !" && (unique_instance == nullptr)));
    unique_instance = std::make_shared<Displayer>(t_width, t_height, scale_factor);
    return unique_instance;
}
// ====================================================================================================================
std::shared_ptr<Displayer> 
Displayer::instance()
{
    assert(("Il faut initialiser l'instance avant tout !" && (unique_instance != nullptr)));
    return unique_instance;
}
