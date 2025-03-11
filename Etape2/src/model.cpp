#include <stdexcept>
#include <cmath>
#include <iostream>
#include <mpi.h>  // 添加 MPI 头文件
#include "model.hpp"

namespace
{
    double pseudo_random(std::size_t index, std::size_t time_step)
    {
        std::uint_fast32_t xi = std::uint_fast32_t(index * (time_step + 1));
        std::uint_fast32_t r = (48271 * xi) % 2147483647;
        return r / 2147483646.;
    }

    double log_factor(std::uint8_t value)
    {
        return std::log(1. + value) / std::log(256);
    }
}

Model::Model(double t_length, unsigned t_discretization, std::array<double, 2> t_wind,
             LexicoIndices t_start_fire_position, double t_max_wind)
    : m_length(t_length),
      m_distance(-1),
      m_geometry(t_discretization),
      m_wind(t_wind),
      m_wind_speed(std::sqrt(t_wind[0] * t_wind[0] + t_wind[1] * t_wind[1])),
      m_max_wind(t_max_wind),
      m_vegetation_map(t_discretization * t_discretization, 255u),
      m_fire_map(t_discretization * t_discretization, 0u)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取进程编号

    if (t_discretization == 0)
    {
        throw std::range_error("Le nombre de cases par direction doit être plus grand que zéro.");
    }
    m_distance = m_length / double(m_geometry);
    auto index = get_index_from_lexicographic_indices(t_start_fire_position);
    
    if (rank == 0) // 仅由进程 0 初始化火灾
    {
        m_fire_map[index] = 255u;
        m_fire_front[index] = 255u;
    }

    constexpr double alpha0 = 4.52790762e-01;
    constexpr double alpha1 = 9.58264437e-04;
    constexpr double alpha2 = 3.61499382e-05;

    if (m_wind_speed < t_max_wind)
        p1 = alpha0 + alpha1 * m_wind_speed + alpha2 * (m_wind_speed * m_wind_speed);
    else
        p1 = alpha0 + alpha1 * t_max_wind + alpha2 * (t_max_wind * t_max_wind);
    p2 = 0.3;
}

bool Model::update()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. 将 unordered_map 转换为 vector
    std::vector<unsigned char> fire_front_buffer(m_geometry * m_geometry, 0);
    for (const auto& [key, value] : m_fire_front)
    {
        fire_front_buffer[key] = value;
    }

    // 2. 进程 0 需要接收所有进程的结果
    std::vector<unsigned char> global_fire_front;
    if (rank == 0)
    {
        global_fire_front.resize(m_geometry * m_geometry * size);
    }

    // 3. 进程间通信，收集数据
    MPI_Gather(fire_front_buffer.data(), fire_front_buffer.size(), MPI_UNSIGNED_CHAR,
               global_fire_front.data(), fire_front_buffer.size(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // 4. 进程 0 需要重新转换 vector 到 unordered_map
    if (rank == 0)
    {
        m_fire_front.clear();
        for (size_t i = 0; i < global_fire_front.size(); i++)
        {
            if (global_fire_front[i] > 0)
            {
                m_fire_front[i] = global_fire_front[i];
            }
        }
    }

    // 5. 更新时间步
    m_time_step += 1;
    return !m_fire_front.empty();
}
std::size_t Model::get_index_from_lexicographic_indices(LexicoIndices t_lexico_indices) const
{
    return t_lexico_indices.row * this->geometry() + t_lexico_indices.column;
}

