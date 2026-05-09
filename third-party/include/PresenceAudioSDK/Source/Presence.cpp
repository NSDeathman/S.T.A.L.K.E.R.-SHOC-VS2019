/*
====================================================================================================
  Presence Audio SDK - Main System Interface
  High-Performance Real-time Audio Path Tracing & EAX Simulation Library
====================================================================================================

  Copyright (c) 2026 Presence Collaboratory, NSDeathman & Gemini 3

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  1. The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

  2. Any project (commercial, free, open-source, or closed-source) using this Software
     must include attribution to "Presence Audio SDK by Presence Collaboratory" in its
     documentation, credits, or about screen.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

====================================================================================================
  Developed by: NSDeathman (Architecture & Core), Gemini 3 (Optimization & Math)
  Organization: Presence Collaboratory
====================================================================================================
*/

#include <vector>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstring>
#include <random>
#include <array>
#include <algorithm>
#include <condition_variable>
#include <unordered_map>
#include <functional>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>

// Parallel Patterns Library (Windows)
#ifdef _WIN32
#   include <ppl.h>
#endif

#include "../Include/PresenceMacros.h"
#include "../Include/PresenceSystem.h"

PRESENCE_BEGIN

// =================================================================================================
// UTILS & CONSTANTS
// =================================================================================================
static const int MAX_TRACKED_MATERIALS = 64;

// Sphere of directions (46 vectors)
static const float3 sphereDirections[] =
{
    { 1,0,0 },
    { -1,0,0 },
    { 0,1,0 },
    { 0,-1,0 },
    { 0,0,1 },
    { 0,0,-1 },
    { 0.577f,0.577f,0.577f },
    { 0.577f,0.577f,-0.577f },
    { 0.577f,-0.577f,0.577f },
    { 0.577f,-0.577f,-0.577f },
    { -0.577f,0.577f,0.577f },
    { -0.577f,0.577f,-0.577f },
    { -0.577f,-0.577f,0.577f },
    { -0.577f,-0.577f,-0.577f },
    { 0,0.525f,0.850f },
    { 0,0.525f,-0.850f },
    { 0,-0.525f,0.850f },
    { 0,-0.525f,-0.850f },
    { 0.850f,0,0.525f },
    { 0.850f,0,-0.525f },
    { -0.850f,0,0.525f },
    { -0.850f,0,-0.525f },
    { 0.525f,0.850f,0 },
    { 0.525f,-0.850f,0 },
    { -0.525f,0.850f,0 },
    { -0.525f,-0.850f,0 }
};
static const int DIRECTIONS_COUNT = sizeof(sphereDirections) / sizeof(float3);

// =================================================================================================
// VERSION INFO
// =================================================================================================
float AudioSystem::GetVersion()
{
    return 0.3f;
}

const char* AudioSystem::GetVersionString()
{
    return "Presence Audio ver. 0.3";
}

// =================================================================================================
// THREAD-SAFE RANDOM GENERATOR
// =================================================================================================
class ThreadSafeRandom
{
private:
    static std::mutex seedMutex;
    static uint32_t nextSeed;
    static thread_local std::unique_ptr<std::mt19937> tlsRng;

public:
    static std::mt19937& Get()
    {
        if (!tlsRng)
        {
            uint32_t seed;
            {
                std::lock_guard<std::mutex> lock(seedMutex);
                seed = nextSeed++ + static_cast<uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count());
            }
            tlsRng = std::make_unique<std::mt19937>(seed);
        }
        return *tlsRng;
    }

    static float3 Direction()
    {
        auto& rng = Get();
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        float3 v;
        do
        {
            v = float3(d(rng), d(rng), d(rng));
        } while (v.length_sq() > 1.0f || v.length_sq() < 0.001f);
        return v.normalize();
    }

    static float3 HemisphereDir(const float3& normal)
    {
        float3 d = Direction();
        if (d.dot(normal) < 0)
            d = d * -1.0f;
        return d;
    }
};

std::mutex ThreadSafeRandom::seedMutex;
uint32_t ThreadSafeRandom::nextSeed = 12345;
thread_local std::unique_ptr<std::mt19937> ThreadSafeRandom::tlsRng;

// =================================================================================================
// PROFILER
// =================================================================================================
class PerformanceProfiler
{
    struct ThreadLocalData
    {
        uint64_t totalCalls = 0;
        uint64_t totalTimeNS = 0;
        uint64_t maxTimeNS = 0;
    };
    using ThreadId = std::thread::id;
    std::unordered_map<std::string, std::unordered_map<ThreadId, ThreadLocalData>> data;
    std::mutex mutex;
    bool enabled = false;

public:
    void SetEnabled(bool enable)
    {
        enabled = enable;
    }

    bool IsEnabled() const
    {
        return enabled;
    }

    class ScopedTimer
    {
        PerformanceProfiler& profiler;
        const char* name;
        std::chrono::high_resolution_clock::time_point start;
        bool active;

    public:
        ScopedTimer(const char* timerName, PerformanceProfiler& prof)
            : profiler(prof)
            , name(timerName)
            , active(profiler.IsEnabled())
        {
            if (active)
                start = std::chrono::high_resolution_clock::now();
        }

        ~ScopedTimer()
        {
            if (active)
            {
                auto end = std::chrono::high_resolution_clock::now();
                profiler.Record(name,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            }
        }
    };

    void Record(const char* timerName, uint64_t ns)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& threadData = data[timerName][std::this_thread::get_id()];
        threadData.totalCalls++;
        threadData.totalTimeNS += ns;
        threadData.maxTimeNS = std::max(threadData.maxTimeNS, ns);
    }
};

#define PROFILE_SCOPE(name) \
    Presence::PerformanceProfiler::ScopedTimer __timer(name, profiler)

// =================================================================================================
// MATERIAL SYSTEM
// =================================================================================================
class MaterialSystem
{
    std::vector<MaterialParams> materials;
    mutable std::shared_mutex mutex;

public:
    MaterialSystem()
    {
        materials.resize(static_cast<int>(MaterialType::Count));
        Set(static_cast<int>(MaterialType::Air),      { 1.00f, 0.00f, 0.00f, 0.00f });
        Set(static_cast<int>(MaterialType::Stone),    { 0.15f, 0.70f, 0.15f, 0.70f });
        Set(static_cast<int>(MaterialType::Metal),    { 0.10f, 0.95f, 0.05f, 0.80f });
        Set(static_cast<int>(MaterialType::Wood),     { 0.40f, 0.30f, 0.40f, 0.30f });
        Set(static_cast<int>(MaterialType::Soft),     { 0.80f, 0.10f, 0.80f, 0.10f });
        Set(static_cast<int>(MaterialType::Glass),    { 0.30f, 0.85f, 0.05f, 0.60f });
        Set(static_cast<int>(MaterialType::Absorber), { 0.00f, 0.01f, 0.99f, 0.01f });
    }

    bool Set(int id, const MaterialParams& p)
    {
        std::unique_lock<std::shared_mutex> lock(mutex);
        if (id >= 0)
        {
            if (id >= static_cast<int>(materials.size()))
                materials.resize(id + 1);
            materials[id] = p;
            return true;
        }
        return false;
    }

    int Add(const MaterialParams& p)
    {
        std::unique_lock<std::shared_mutex> lock(mutex);
        materials.push_back(p);
        return static_cast<int>(materials.size()) - 1;
    }

    MaterialParams Get(int id) const
    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        if (id < 0 || id >= static_cast<int>(materials.size()))
            return materials[0];
        return materials[id];
    }

    float GetStepSize(int id) const
    {
        return (Get(id).transmission > 0.5f) ? 0.2f : 0.4f;
    }
};

// =================================================================================================
// OCCLUSION CALCULATOR
// =================================================================================================
class OcclusionCalculator
{
public:
    struct Config
    {
        float maxDistance = 80.0f;
    };

private:
    IGeometryProvider* provider;
    const MaterialSystem& materials;
    PerformanceProfiler& profiler;
    Config config;

    struct CacheEntry
    {
        float occlusion;
        uint64_t lastFrame;
        float3 lPos;
        float3 sPos;
    };
    mutable std::unordered_map<uint64_t, CacheEntry> cache;
    mutable std::unordered_map<uint64_t, float> history;
    mutable std::mutex cacheMutex;
    uint64_t currentFrame = 0;

public:
    OcclusionCalculator(IGeometryProvider* p, const MaterialSystem& m, PerformanceProfiler& prof)
        : provider(p)
        , materials(m)
        , profiler(prof)
    {
    }

    void SetConfig(const Config& c)
    {
        config = c;
    }

    void Tick()
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        currentFrame++;
        if (currentFrame % 1000 == 0)
        {
            for (auto it = cache.begin(); it != cache.end(); )
            {
                if (currentFrame - it->second.lastFrame > 30)
                    it = cache.erase(it);
                else
                    ++it;
            }
            if (history.size() > 2000)
                history.clear();
        }
    }

    float Calculate(const float3& listenerPos, const float3& sourcePos) const
    {
        float3 dir = sourcePos - listenerPos;
        float dist = dir.magnitude();
        if (dist < 0.5f)
            return 1.0f;
        if (dist > config.maxDistance)
            return 0.0f;
        dir = dir.normalize();

        RayHit hit = provider->CastRay(listenerPos, dir, dist);
        if (!hit.isHit)
            return 1.0f;

        float energy = 1.0f;
        float trans = materials.Get(hit.materialID).transmission;
        energy *= trans;

        if (energy > 0.1f)
        {
            float3 p = listenerPos;
            p.mad(dir, hit.distance + 0.5f);
            float remainingDist = dist - hit.distance - 0.5f;
            if (remainingDist > 0)
            {
                RayHit h2 = provider->CastRay(p, dir, remainingDist);
                if (h2.isHit)
                    energy *= materials.Get(h2.materialID).transmission;
            }
        }
        return std::max(energy, 0.0f);
    }
};

// =================================================================================================
// MAIN SYSTEM IMPL
// =================================================================================================
class AudioSystem::Impl
{
public:
    IGeometryProvider* provider = nullptr;
    Settings settings;
    PerformanceProfiler profiler;
    MaterialSystem materials;

    std::unique_ptr<OcclusionCalculator> occlusion;
    std::unique_ptr<std::thread> workerThread;

    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> stopThread{ false };
    std::atomic<bool> workPending{ false };
    std::atomic<bool> initialized{ false };

    float3 targetPos;
    float3 lastPos;
    float fog = 0.0f;

    struct AnalysisContext
    {
        float3 cameraPosition;
        int frameCount = 0;

        // New metrics for accuracy
        float rawTotalDist = 0.0f;   // Unweighted geometric distance sum
        int totalSegments = 0;       // Total number of path segments traced

        // Old energy-weighted metrics (for reverb tail calc)
        float energyWeightedDist = 0.0f;

        // Openness metric
        int firstHitCount = 0;       // How many rays hit something immediately

        int materialHits[MAX_TRACKED_MATERIALS] = { 0 };

        // Results
        float physicalVolume = 100.0f;
        float meanFreePath = 5.0f;
        float physicalReflectivity = 0.3f;
        float geometricOpenness = 0.5f;
        float geometricEnclosedness = 0.5f;
        float reverbTime = 1.0f;
        float perceivedReflectivity = 0.0f;

        void Reset()
        {
            frameCount = 0;
            rawTotalDist = 0;
            totalSegments = 0;
            energyWeightedDist = 0;
            firstHitCount = 0;
            std::memset(materialHits, 0, sizeof(materialHits));
        }

        static float Lerp(float a, float b, float t)
        {
            return a + (b - a) * t;
        }

        static float Clamp(float v, float minVal, float maxVal)
        {
            return (v < minVal) ? minVal : (v > maxVal) ? maxVal : v;
        }
    } context;

    EAXResult targetEAX;
    EAXResult currentEAX;

    ~Impl()
    {
        Shutdown();
    }

    void Init(IGeometryProvider* p, const Settings& s)
    {
        provider = p;
        settings = s;
        occlusion = std::make_unique<OcclusionCalculator>(p, materials, profiler);
        initialized = true;
        context.Reset();
        if (settings.useMultithreading)
            workerThread = std::make_unique<std::thread>(&Impl::PhysicsLoop, this);
    }

    void Shutdown()
    {
        initialized = false;
        if (workerThread)
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopThread = true;
            }
            cv.notify_all();
            if (workerThread->joinable())
                workerThread->join();
            workerThread.reset();
        }
    }

    // ---------------------------------------------------------------------------------------------
    // MATH: CalculatePhysics (Improved MFP)
    // ---------------------------------------------------------------------------------------------
    void CalculatePhysics()
    {
        float numRays = static_cast<float>(DIRECTIONS_COUNT);
        float totalFrames = std::max(static_cast<float>(context.frameCount), 1.0f);

        // 1. Mean Free Path (MFP)
        // Use average segment length, not total path length
        float totalSegments = std::max(static_cast<float>(context.totalSegments), 1.0f);
        float avgSegment = context.rawTotalDist / totalSegments;
        context.meanFreePath = std::max(avgSegment, 0.5f);

        // 2. Geometric Enclosedness (ratio of rays that hit something vs sky)
        float totalFirstHits = static_cast<float>(context.firstHitCount) / totalFrames;
        float hitRatio = totalFirstHits / numRays;
        context.geometricEnclosedness = AnalysisContext::Clamp(hitRatio, 0.0f, 1.0f);
        context.geometricOpenness = 1.0f - context.geometricEnclosedness;

        // 3. Physical Volume
        // V = 4/3 * PI * R^3. Using MFP as radius gives a good approximation for convex rooms.
        // For a 10x10x4 room, MFP ~ 4.5m. V ~ 380m^3. Size ~ 7.2m.
        float estVol = (4.0f / 3.0f) * PI * std::pow(context.meanFreePath, 3.0f);
        if (context.geometricOpenness > 0.6f)
            estVol = std::min(estVol, 50000.0f);
        context.physicalVolume = AnalysisContext::Clamp(estVol, 10.0f, 500000.0f);

        // 4. Physical Reflectivity
        float totalMaterialHits = 0;
        float accumRefl = 0;
        for (int i = 0; i < MAX_TRACKED_MATERIALS; ++i)
        {
            if (context.materialHits[i] > 0)
            {
                float hits = static_cast<float>(context.materialHits[i]) / totalFrames;
                totalMaterialHits += hits;
                accumRefl += hits * materials.Get(i).rt60_weight;
            }
        }
        float baseReflectivity = (totalMaterialHits > 0) ? (accumRefl / totalMaterialHits) : 0.1f;
        float effectiveReflectivity = baseReflectivity *
            (1.0f - context.geometricOpenness * 0.8f);
        context.physicalReflectivity = AnalysisContext::Clamp(effectiveReflectivity, 0.05f, 0.95f);
        context.perceivedReflectivity = std::pow(context.physicalReflectivity, 0.5f);

        // 5. Reverb Time (Eyring)
        float surfArea = 6.0f * std::pow(context.physicalVolume, 2.0f / 3.0f);
        float absCoeff = AnalysisContext::Clamp(1.0f - context.physicalReflectivity, 0.01f, 0.99f);
        float rt60 = 0.161f * context.physicalVolume /
            (-surfArea * std::log(1.0f - absCoeff));

        // Tweaks
        if (context.physicalVolume < 150.0f)
            rt60 *= std::max(0.3f, context.physicalVolume / 150.0f);
        if (context.geometricOpenness > 0.4f)
        {
            rt60 *= AnalysisContext::Clamp(
                1.0f - (context.geometricOpenness - 0.4f) * 1.8f,
                0.1f,
                1.0f
            );
        }
        context.reverbTime = AnalysisContext::Clamp(rt60, 0.1f, 8.0f);
    }

    void ComputeEAX()
    {
        CalculatePhysics();

        EAXResult eax;
        eax.isValid = true;
        eax.debugEnclosedness = context.geometricEnclosedness;
        eax.debugOpenness = context.geometricOpenness;

        float fogVal = AnalysisContext::Clamp(fog, 0.0f, 1.0f);
        float open = context.geometricOpenness;

        // Room
        float roomInt = 0.3f + (0.7f * context.geometricEnclosedness);
        roomInt *= std::sqrt(context.perceivedReflectivity);
        if (open > 0.4f)
        {
            float t = AnalysisContext::Clamp((open - 0.4f) / 0.6f, 0.0f, 1.0f);
            roomInt *= AnalysisContext::Clamp(1.0f - (t * t), 0.1f, 1.0f);
        }
        float val = AnalysisContext::Lerp(-4000.0f, 400.0f,
            AnalysisContext::Clamp(roomInt - (fogVal * 0.4f), 0.0f, 1.1f));
        eax.lRoom = static_cast<int32_t>(val);
        if (context.physicalVolume < 70.0f && context.geometricEnclosedness > 0.8f)
            eax.lRoom += 500;
        eax.lRoom = static_cast<int32_t>(
            AnalysisContext::Clamp(static_cast<float>(eax.lRoom), -10000.0f, 600.0f));

        // Decay
        eax.flDecayTime = context.reverbTime;
        if (fogVal > 0.5f)
            eax.flDecayTime *= 0.8f;

        // Reflections
        float reflScale = (open > 0.6f) ? 0.8f : 1.0f;
        float reflectionsVal = AnalysisContext::Lerp(-2500.0f, 200.0f,
            context.physicalReflectivity * reflScale);
        eax.lReflections = static_cast<int32_t>(
            AnalysisContext::Clamp(reflectionsVal, -10000.0f, 500.0f));

        float baseDelay = context.meanFreePath / 340.0f; // MFP is now accurate (~5m -> 0.015s)
        if (open > 0.6f)
            baseDelay *= 1.5f;
        eax.flReflectionsDelay = AnalysisContext::Clamp(baseDelay, 0.0f, 0.3f);

        // Reverb
        float reverbVal = AnalysisContext::Lerp(-3000.0f, 0.0f, roomInt);
        if (open > 0.7f)
            reverbVal -= 600.0f;
        eax.lReverb = static_cast<int32_t>(
            AnalysisContext::Clamp(reverbVal, -10000.0f, 200.0f));
        eax.flReverbDelay = AnalysisContext::Clamp(eax.flReflectionsDelay + 0.02f, 0.0f, 0.1f);

        // Tone
        float hfLoss = -100.0f;
        float outdoorCut = open * -1200.0f;
        float roomHFVal = static_cast<float>(eax.lRoom) + hfLoss + outdoorCut;
        eax.lRoomHF = static_cast<int32_t>(
            AnalysisContext::Clamp(roomHFVal, -10000.0f, -100.0f));

        eax.flDecayHFRatio = (open > 0.6f) ? 0.5f : 0.83f;
        eax.flEnvironmentDiffusion = (open > 0.7f) ? 0.6f : 1.0f;
        eax.flAirAbsorptionHF = -5.0f + (open * -8.0f) - (fogVal * 5.0f);

        // Env Size: side length of the acoustic space (Volume^(1/3))
        eax.flEnvironmentSize = std::pow(context.physicalVolume, 1.0f / 3.0f);

        {
            std::lock_guard<std::mutex> lock(mutex);
            targetEAX = eax;
        }
    }

    void PhysicsLoop()
    {
        while (!stopThread)
        {
            float3 curPos;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return workPending || stopThread; });
                if (stopThread)
                    break;
                curPos = targetPos;
                workPending = false;
            }

            if (lastPos.distance_to(curPos) > 0.5f)
            {
                context.Reset();
                lastPos = curPos;
            }
            context.cameraPosition = curPos;

            {
                PROFILE_SCOPE("EAX::Trace");
                for (int i = 0; i < DIRECTIONS_COUNT; ++i)
                {
                    float energy = 1.0f;
                    float3 pos = curPos;
                    float3 dir = sphereDirections[i];

                    for (int b = 0; b < settings.maxBounces; ++b)
                    {
                        RayHit hit = provider->CastRay(pos, dir, settings.maxRayDistance);

                        // Hit nothing (sky)
                        if (!hit.isHit)
                        {
                            // Treat sky as a segment of max length
                            context.rawTotalDist += settings.maxRayDistance;
                            context.totalSegments++;
                            context.energyWeightedDist += settings.maxRayDistance * energy;
                            break;
                        }

                        // Hit wall
                        // 1. Geometric data (unweighted)
                        context.rawTotalDist += hit.distance;
                        context.totalSegments++;

                        // 2. Acoustic data (weighted)
                        context.energyWeightedDist += hit.distance * energy;

                        // 3. Openness check
                        if (b == 0)
                            context.firstHitCount++;

                        // 4. Material
                        if (hit.materialID >= 0 && hit.materialID < MAX_TRACKED_MATERIALS)
                            context.materialHits[hit.materialID]++;

                        energy *= (1.0f - materials.Get(hit.materialID).absorption);
                        if (energy < 0.05f)
                            break;

                        pos.mad(dir, hit.distance - 0.05f);
                        dir = ThreadSafeRandom::HemisphereDir(hit.normal);
                    }
                }
            }

            context.frameCount++;
            ComputeEAX();
            if (occlusion)
                occlusion->Tick();
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
};

// =================================================================================================
// PUBLIC API
// =================================================================================================

AudioSystem::AudioSystem()
    : m_Impl(new Impl())
{
}

AudioSystem::~AudioSystem()
{
    Shutdown();
    delete m_Impl;
}

void AudioSystem::Initialize(IGeometryProvider* p, const Settings& s)
{
    if (m_Impl)
        m_Impl->Init(p, s);
}

void AudioSystem::Shutdown()
{
    if (m_Impl)
        m_Impl->Shutdown();
}

template <typename T>
static T LerpEAX(T current, T target, float step)
{
    float diff = static_cast<float>(target - current);
    if (std::abs(diff) < 0.1f)
        return target;
    return static_cast<T>(static_cast<float>(current) + diff * step);
}

void AudioSystem::Update(const float3& pos, float dt, float fogVal)
{
    if (!m_Impl || !m_Impl->initialized)
        return;
    m_Impl->fog = fogVal;

    if (m_Impl->settings.useMultithreading)
    {
        {
            std::lock_guard<std::mutex> lock(m_Impl->mutex);
            m_Impl->targetPos = pos;
            m_Impl->workPending = true;
        }
        m_Impl->cv.notify_one();
    }

    EAXResult target;
    {
        std::lock_guard<std::mutex> lock(m_Impl->mutex);
        target = m_Impl->targetEAX;
    }

    if (target.isValid)
    {
        float speed = 2.0f * dt;
        if (std::abs(target.lRoom - m_Impl->currentEAX.lRoom) > 1000)
            speed = 5.0f * dt;

        auto& curr = m_Impl->currentEAX;
        curr.lRoom = LerpEAX(curr.lRoom, target.lRoom, speed);
        curr.lRoomHF = LerpEAX(curr.lRoomHF, target.lRoomHF, speed);
        curr.lReflections = LerpEAX(curr.lReflections, target.lReflections, speed);
        curr.lReverb = LerpEAX(curr.lReverb, target.lReverb, speed);
        curr.flDecayTime = LerpEAX(curr.flDecayTime, target.flDecayTime, speed);
        curr.flEnvironmentSize = LerpEAX(curr.flEnvironmentSize, target.flEnvironmentSize, speed);

        curr.flDecayHFRatio = target.flDecayHFRatio;
        curr.flReflectionsDelay = target.flReflectionsDelay;
        curr.flReverbDelay = target.flReverbDelay;
        curr.flAirAbsorptionHF = target.flAirAbsorptionHF;
        curr.flEnvironmentDiffusion = target.flEnvironmentDiffusion;
        curr.isValid = true;
    }
}

EAXResult AudioSystem::GetEAXResult() const
{
    return m_Impl ? m_Impl->currentEAX : EAXResult();
}

bool AudioSystem::SetMaterialProperties(int id, const MaterialParams& p)
{
    return m_Impl && m_Impl->materials.Set(id, p);
}

int AudioSystem::CreateCustomMaterial(const MaterialParams& p)
{
    return m_Impl ? m_Impl->materials.Add(p) : -1;
}

float AudioSystem::CalculateOcclusion(const float3& listenerPos, const float3& sourcePos)
{
    if (!m_Impl || !m_Impl->occlusion)
        return 1.0f;
    return m_Impl->occlusion->Calculate(listenerPos, sourcePos);
}

PRESENCE_END
