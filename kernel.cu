﻿//--------------------------------------------------------------------------------------------------------------------------------------------------
// Project : Real-time-Particle-Simulation-with-CUDA
// Implement a Particle Simulation with CUDA acceleration with real-time visualization.
// Author: Arsheya Raj
// Date: 4th April 2025
//--------------------------------------------------------------------------------------------------------------------------------------------------
//
//	Develope a system that uses CUDA to accelerate the simulation of particle systems. This enables real-time visualization and
//  analysis of complex fluid dynamics or physics simulations on local hardware.
// 
//--------------------------------------------------------------------------------------------------------------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <sstream>

const int NUM_PARTICLES = 10000;    // Increased number of particles
const int NUM_FRAMES = 1000;        // Increased number of frames
const float DT = 0.01f;
const float GRAVITY = -9.81f;
const float ELASTICITY = 0.9f;

// Waterfall effect parameters
const float MAX_X_SPREAD = 5.0f;  // Max horizontal spread in X direction
const float MAX_Z_SPREAD = 5.0f;  // Max horizontal spread in Z direction
const float INIT_Y = 20.0f;       // Starting Y position (top of the waterfall)

// Particle structure
struct Particle {
    float3 position;
    float3 velocity;
};

// CUDA kernel for updating particle positions
__global__ void updateParticles(Particle* particles, int numParticles, float dt, float gravity, float elasticity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        Particle& p = particles[idx];

        // Apply gravity to the velocity
        p.velocity.y += gravity * dt;

        // Update particle position based on velocity
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
        p.position.z += p.velocity.z * dt;

        // Bounce particles off the ground with elasticity
        if (p.position.y < 0) {
            p.position.y = 0;
            p.velocity.y *= -elasticity;
        }
    }
}

// Helper function to simulate particles with CUDA
cudaError_t simulateParticlesWithCuda(Particle* particles, int numParticles, float dt, float gravity, float elasticity) {
    Particle* d_particles = nullptr;

    cudaError_t cudaStatus = cudaMalloc((void**)&d_particles, numParticles * sizeof(Particle));
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMemcpy(d_particles, particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    // Launch kernel to update particles
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    updateParticles << <numBlocks, blockSize >> > (d_particles, numParticles, dt, gravity, elasticity);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMemcpy(particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaFree(d_particles);
    return cudaSuccess;
}

// Function to append particle data to a CSV file
void appendParticlesToCSV(const Particle* particles, int numParticles, int frame, std::ofstream& file) {
    for (int i = 0; i < numParticles; i++) {
        file << frame << "," << i << ","
            << particles[i].position.x << ","
            << particles[i].position.y << ","
            << particles[i].position.z << "\n";
    }
}

int main() {
    Particle particles[NUM_PARTICLES];
    srand((unsigned int)time(0));

    // Initialize particles with random positions for waterfall effect
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].position = make_float3(
            (rand() % 2 == 0 ? 1 : -1) * (rand() % (int)(MAX_X_SPREAD * 2)),  // Random X position (spread in range [-MAX_X_SPREAD, MAX_X_SPREAD])
            INIT_Y,  // Y starts at top (higher)
            (rand() % 2 == 0 ? 1 : -1) * (rand() % (int)(MAX_Z_SPREAD * 2))   // Random Z position (spread in range [-MAX_Z_SPREAD, MAX_Z_SPREAD])
        );

        particles[i].velocity = make_float3(
            rand() % 10 - 5,  // Random horizontal velocity
            0.0f,             // Initial Y velocity is 0 (falling due to gravity)
            rand() % 10 - 5   // Random Z velocity
        );
    }

    // Open CSV file to save particle positions
    std::ofstream file("particles.csv");
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open CSV file!\n");
        return 1;
    }
    file << "frame,particle_id,x,y,z\n";

    // Simulate and record the positions of particles over time
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        cudaError_t cudaStatus = simulateParticlesWithCuda(particles, NUM_PARTICLES, DT, GRAVITY, ELASTICITY);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "simulateParticlesWithCuda failed at frame %d!\n", frame);
            return 1;
        }

        // Save particle positions to CSV
        appendParticlesToCSV(particles, NUM_PARTICLES, frame, file);
        printf("Saved frame %d\n", frame);
    }

    // Close the CSV file
    file.close();
    return 0;
}
