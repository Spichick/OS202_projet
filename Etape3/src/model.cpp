#include <stdexcept>
#include <cmath>
#include <iostream>
#include <chrono>
#include "model.hpp"
#include <omp.h>
#include <mpi.h>
double m_total_time = 0.0; // 总时间
double m_step_count = 0; // 时间步数
namespace
{
    double pseudo_random( std::size_t index, std::size_t time_step )
    {
        std::uint_fast32_t xi = std::uint_fast32_t(index*(time_step+1));
        std::uint_fast32_t r  = (48271*xi)%2147483647;
        return r/2147483646.;
    }

    double log_factor( std::uint8_t value )
    {
        return std::log(1.+value)/std::log(256);
    }
}

Model::Model( double t_length, unsigned t_discretization, std::array<double,2> t_wind,
              LexicoIndices t_start_fire_position, double t_max_wind )
    :   m_length(t_length),
        m_distance(-1),
        m_geometry(t_discretization),
        m_wind(t_wind),
        m_wind_speed(std::sqrt(t_wind[0]*t_wind[0] + t_wind[1]*t_wind[1])),
        m_max_wind(t_max_wind),
        m_vegetation_map(t_discretization*t_discretization, 255u),
        m_fire_map(t_discretization*t_discretization, 0u)

{
    if (t_discretization == 0)
    {
        throw std::range_error("Le nombre de cases par direction doit être plus grand que zéro.");
    }
    m_distance = m_length/double(m_geometry);
    auto index = get_index_from_lexicographic_indices(t_start_fire_position);
    m_fire_map[index] = 255u;
    m_fire_front[index] = 255u; // 在h文件里定义，是uint8_t类型，这里255u的u是unsigned，取值范围就是0到255。值表示火势大小。

    constexpr double alpha0 = 4.52790762e-01;
    constexpr double alpha1 = 9.58264437e-04;
    constexpr double alpha2 = 3.61499382e-05;

    if (m_wind_speed < t_max_wind)
        p1 = alpha0 + alpha1*m_wind_speed + alpha2*(m_wind_speed*m_wind_speed);
    else 
        p1 = alpha0 + alpha1*t_max_wind + alpha2*(t_max_wind*t_max_wind);
    p2 = 0.3; // 火焰减小概率

    if (m_wind[0] > 0)
    {
        alphaEastWest = std::abs(m_wind[0]/t_max_wind)+1;
        alphaWestEast = 1.-std::abs(m_wind[0]/t_max_wind);    
    }
    else
    {
        alphaWestEast = std::abs(m_wind[0]/t_max_wind)+1;
        alphaEastWest = 1. - std::abs(m_wind[0]/t_max_wind);
    }

    if (m_wind[1] > 0)
    {
        alphaSouthNorth = std::abs(m_wind[1]/t_max_wind) + 1;
        alphaNorthSouth = 1. - std::abs(m_wind[1]/t_max_wind);
    }
    else
    {
        alphaNorthSouth = std::abs(m_wind[1]/t_max_wind) + 1;
        alphaSouthNorth = 1. - std::abs(m_wind[1]/t_max_wind);
    }
}
// --------------------------------------------------------------------------------------------------------------------
bool Model::update() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto start_time = std::chrono::high_resolution_clock::now();
    int get_num = omp_get_max_threads();
    // 线程本地存储的临时容器，用于收集每个线程的更新
    std::vector<std::unordered_map<std::size_t, std::uint8_t>> currthread_nextfront(get_num); //TODO: 一个哈希表: 局部计算结果，但不共享数据。适用于快速查找
    std::vector<std::vector<std::size_t>> currthread_erasedkey(get_num);
    std::unordered_map<std::size_t, std::uint8_t> next_front;
    // 并行处理每个火点
    #pragma omp parallel for schedule(dynamic, 10) // 每次分配10个任务给某个线程
    for (size_t i = rank; i < m_fire_front.size(); i+=size) { //对于mpi
        int curr_id = omp_get_thread_num();  // 当前线程index
        auto& local_next_front = currthread_nextfront[curr_id];  // 储存本地新点燃的火点索引
        auto& local_erased = currthread_erasedkey[curr_id];      // 记录火势熄灭的单元索引

        auto f = *(std::next(m_fire_front.begin(), i)); // 获取第i个元素的迭代器，是一个键值对如{43,255}
        // 获取当前火点的坐标
        LexicoIndices coord = get_lexicographic_from_index(f.first);
        
        // 获取当前火点的强度
        double power = log_factor(f.second);

        // 检查并更新相邻单元格
        if (coord.row < m_geometry - 1) {
            double tirage = pseudo_random(f.first + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first + m_geometry];
            double correction = power * log_factor(green_power);
            if (tirage < alphaSouthNorth * p1 * correction) {
                m_fire_map[f.first + m_geometry] = 255;  // 更新火点
                local_next_front[f.first + m_geometry] = 255;  // 本地记录新增火点
            }
        }

        if (coord.row > 0) {
            double tirage = pseudo_random(f.first * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first - m_geometry];
            double correction = power * log_factor(green_power);
            if (tirage < alphaNorthSouth * p1 * correction) {
                m_fire_map[f.first - m_geometry] = 255;
                local_next_front[f.first - m_geometry] = 255;
            }
        }

        if (coord.column < m_geometry - 1) {
            double tirage = pseudo_random(f.first * 13427 * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first + 1];
            double correction = power * log_factor(green_power);
            if (tirage < alphaEastWest * p1 * correction) {
                m_fire_map[f.first + 1] = 255;
                local_next_front[f.first + 1] = 255;
            }
        }

        if (coord.column > 0) {
            double tirage = pseudo_random(f.first * 13427 * 13427 * 13427 + m_time_step, m_time_step);
            double green_power = m_vegetation_map[f.first - 1];
            double correction = power * log_factor(green_power);
            if (tirage < alphaWestEast * p1 * correction) {
                m_fire_map[f.first - 1] = 255;
                local_next_front[f.first - 1] = 255;
            }
        }

        // 处理火势减弱
        if (f.second == 255) {
            double tirage = pseudo_random(f.first * 52513 + m_time_step, m_time_step);
            if (tirage < p2) 
            {
                // 让本点火势减半
                // 再让将要传播到的邻居点火势减半
                m_fire_map[f.first] >>= 1; // >>=1 就是减半，相当于二进制右移一位
                f.second = (f.second >> 1);
                local_next_front[f.first] = f.second;  // 火势减弱
            } 
            // NOTE: Gpt m'a aidé.
            // Dans la version parallèle, comme local_next_front est utilisé, l'affectation dans else est indispensable, sinon certains foyers pourraient ne pas être enregistrés dans m_fire_front final, ce qui entraînerait une perte de foyers.
            else 
            {
                local_next_front[f.first] = f.second;  // 火势保持不变
            }
        } 
        else 
        {
            m_fire_map[f.first] >>= 1;
            local_next_front[f.first] = f.second >> 1;  // 火势继续减弱
        }

        // 如果火势减弱到0，记录待删除的键
        if (local_next_front[f.first] == 0) 
        {
            local_erased.push_back(f.first);
        }
    }

    for (auto& local_front : currthread_nextfront) {
        for (auto& [key, val] : local_front) {
            next_front[key] = val;
        }
    }

    for (auto& local_erased : currthread_erasedkey) {
        for (auto key : local_erased) {
            next_front.erase(key);
            m_fire_map[key] = 0;
        }
    }

    m_fire_front = next_front;

    for (auto f : m_fire_front) {
        if (m_vegetation_map[f.first] > 0) {
            m_vegetation_map[f.first] -= 1;
        }
    }
    // 更新时间步
    #pragma omp atomic
    m_time_step += 1;

    // 输出执行时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Temps pour une étape : " << elapsed.count() << " secondes" << std::endl;

    double step_time = elapsed.count();
    #pragma omp atomic
    m_total_time += step_time;
    #pragma omp atomic
    m_step_count+=1;
    double average_time = m_total_time / m_step_count;
    std::cout<<"avg step time: "<<average_time<<std::endl;
    // 返回是否还有活跃火点
    return !m_fire_front.empty();
}
// ====================================================================================================================
std::size_t   
Model::get_index_from_lexicographic_indices( LexicoIndices t_lexico_indices  ) const
{
    return t_lexico_indices.row*this->geometry() + t_lexico_indices.column;
}
// --------------------------------------------------------------------------------------------------------------------
auto 
Model::get_lexicographic_from_index( std::size_t t_global_index ) const -> LexicoIndices
{
    LexicoIndices ind_coords;
    ind_coords.row    = t_global_index/this->geometry();
    ind_coords.column = t_global_index%this->geometry();
    return ind_coords;
}