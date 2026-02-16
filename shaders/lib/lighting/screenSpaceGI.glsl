/*
    Screen-Space Global Illumination (SSGI) / Ray-Traced Indirect Lighting
    Approximates path-traced indirect diffuse lighting using screen-space ray marching.
    Traces rays from each fragment in random hemisphere directions to gather
    indirect light bounces from nearby surfaces visible on screen.
*/

vec3 CosWeightedHemisphereDir(vec3 normal, float xi1, float xi2) {
    float r = sqrt(xi1);
    float phi = 6.28318530718 * xi2;

    vec3 sampleDir = vec3(
        r * cos(phi),
        r * sin(phi),
        sqrt(max(1.0 - xi1, 0.0))
    );

    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    return tangent * sampleDir.x + bitangent * sampleDir.y + normal * sampleDir.z;
}

#ifdef SSGI_ENABLED
float DoScreenSpaceGlobalIllumination(
    vec3 viewPos, vec3 normalM, float z0, float linearZ0, float dither,
    out vec3 giColor
) {
    giColor = vec3(0.0);

    if (z0 < 0.56 || z0 > 0.9999) return 0.0; // 0.56 = hand/entity depth cutoff used throughout the pack

    #if RT_SUNLIGHT_QUALITY == 1
        const int samples = 2;
        const int steps = 6;
    #elif RT_SUNLIGHT_QUALITY == 2
        const int samples = 3;
        const int steps = 8;
    #else
        const int samples = 4;
        const int steps = 12;
    #endif

    float radius = RT_GI_RADIUS;
    float invSteps = 1.0 / float(steps);
    float thicknessThreshold = radius * 0.15 * invSteps;
    float totalWeight = 0.0;
    vec3 totalGI = vec3(0.0);

    for (int i = 0; i < samples; i++) {
        float xi1 = fract(dither + float(i) * 0.618033988);
        float xi2 = fract(dither * 1.414 + float(i) * 0.381966);

        vec3 sampleDir = CosWeightedHemisphereDir(normalM, xi1, xi2);
        vec3 viewDir = mat3(gbufferModelView) * sampleDir;
        float viewDirLen = length(viewDir);
        if (viewDirLen < 0.001) continue;
        viewDir /= viewDirLen;

        vec3 rayStep = viewDir * (radius * invSteps);
        vec3 rayPos = viewPos + rayStep;

        for (int j = 1; j <= steps; j++) {
            vec4 projPos = gbufferProjection * vec4(rayPos, 1.0);
            vec3 screenPos = projPos.xyz / projPos.w * 0.5 + 0.5;

            if (clamp(screenPos.xy, vec2(0.0), vec2(1.0)) != screenPos.xy) break;

            float sampleDepth = texelFetch(depthtex0, ivec2(screenPos.xy * vec2(viewWidth, viewHeight)), 0).r;
            float sampleLinear = (2.0 * near) / (far + near - sampleDepth * farMinusNear);
            float rayLinear = (2.0 * near) / (far + near - screenPos.z * farMinusNear);

            float depthDiff = rayLinear - sampleLinear;

            if (depthDiff > 0.0 && depthDiff < thicknessThreshold) {
                vec3 hitColor = texelFetch(colortex0, ivec2(screenPos.xy * vec2(viewWidth, viewHeight)), 0).rgb;
                float dist = float(j) * invSteps;
                float falloff = 1.0 - dist * dist;
                totalGI += hitColor * falloff;
                totalWeight += falloff;
                break;
            }

            rayPos += rayStep;
        }
    }

    if (totalWeight > 0.0) {
        giColor = totalGI / totalWeight;
        giColor = min(giColor, vec3(2.0));
        return 1.0;
    }

    return 0.0;
}
#endif

#ifdef RT_SHADOW_ENABLED
float DoScreenSpaceShadow(
    vec3 viewPos, vec3 viewLightDir, float z0, float linearZ0, float dither
) {
    if (z0 < 0.56 || z0 > 0.9999) return 1.0;

    #if RT_SHADOW_QUALITY == 1
        const int steps = 6;
    #elif RT_SHADOW_QUALITY == 2
        const int steps = 10;
    #else
        const int steps = 16;
    #endif

    float radius = 3.0;
    float invSteps = 1.0 / float(steps);
    float thicknessThreshold = radius * 0.2 * invSteps;

    float viewDirLen = length(viewLightDir);
    if (viewDirLen < 0.001) return 1.0;
    vec3 normLightDir = viewLightDir / viewDirLen;

    vec3 rayStep = normLightDir * (radius * invSteps);
    vec3 rayPos = viewPos + rayStep * (1.0 + dither * 0.5);

    for (int j = 1; j <= steps; j++) {
        vec4 projPos = gbufferProjection * vec4(rayPos, 1.0);
        vec3 screenPos = projPos.xyz / projPos.w * 0.5 + 0.5;

        if (clamp(screenPos.xy, vec2(0.0), vec2(1.0)) != screenPos.xy) break;

        float sampleDepth = texelFetch(depthtex0, ivec2(screenPos.xy * vec2(viewWidth, viewHeight)), 0).r;
        float sampleLinear = (2.0 * near) / (far + near - sampleDepth * farMinusNear);
        float rayLinear = (2.0 * near) / (far + near - screenPos.z * farMinusNear);

        float depthDiff = rayLinear - sampleLinear;

        if (depthDiff > 0.0 && depthDiff < thicknessThreshold) {
            return mix(1.0, 0.0, RT_SHADOW_STRENGTH * 0.01);
        }

        rayPos += rayStep;
    }

    return 1.0;
}
#endif

#ifdef RT_REFLECTION_ENABLED
vec3 DoScreenSpaceReflection(
    vec3 viewPos, vec3 normalM, vec3 nViewPos, float z0, float linearZ0, float dither,
    out float hitResult
) {
    hitResult = 0.0;

    if (z0 < 0.56 || z0 > 0.9999) return vec3(0.0);

    #if RT_REFLECTION_QUALITY == 1
        const int steps = 8;
    #elif RT_REFLECTION_QUALITY == 2
        const int steps = 16;
    #else
        const int steps = 24;
    #endif

    float radius = 6.0;
    float invSteps = 1.0 / float(steps);
    float thicknessThreshold = radius * 0.15 * invSteps;

    vec3 reflectDir = reflect(nViewPos, normalM);
    vec3 rayStep = reflectDir * (radius * invSteps);
    vec3 rayPos = viewPos + rayStep * (1.0 + dither * 0.25);

    for (int j = 1; j <= steps; j++) {
        vec4 projPos = gbufferProjection * vec4(rayPos, 1.0);
        vec3 screenPos = projPos.xyz / projPos.w * 0.5 + 0.5;

        if (clamp(screenPos.xy, vec2(0.0), vec2(1.0)) != screenPos.xy) break;

        float sampleDepth = texelFetch(depthtex0, ivec2(screenPos.xy * vec2(viewWidth, viewHeight)), 0).r;
        float sampleLinear = (2.0 * near) / (far + near - sampleDepth * farMinusNear);
        float rayLinear = (2.0 * near) / (far + near - screenPos.z * farMinusNear);

        float depthDiff = rayLinear - sampleLinear;

        if (depthDiff > 0.0 && depthDiff < thicknessThreshold) {
            vec3 hitColor = texelFetch(colortex0, ivec2(screenPos.xy * vec2(viewWidth, viewHeight)), 0).rgb;
            float dist = float(j) * invSteps;
            float falloff = 1.0 - dist * dist;
            hitResult = falloff;
            return hitColor;
        }

        rayPos += rayStep;
    }

    return vec3(0.0);
}
#endif

#ifdef RT_AO_ENABLED
float DoScreenSpaceRTAO(
    vec3 viewPos, vec3 normalM, float z0, float linearZ0, float dither
) {
    if (z0 < 0.56 || z0 > 0.9999) return 1.0;

    #if RT_AO_QUALITY == 1
        const int samples = 2;
        const int steps = 4;
    #elif RT_AO_QUALITY == 2
        const int samples = 3;
        const int steps = 6;
    #else
        const int samples = 4;
        const int steps = 8;
    #endif

    float radius = RT_AO_RADIUS;
    float invSteps = 1.0 / float(steps);
    float thicknessThreshold = radius * 0.2 * invSteps;
    float occlusion = 0.0;
    float totalWeight = 0.0;

    for (int i = 0; i < samples; i++) {
        float xi1 = fract(dither + float(i) * 0.618033988);
        float xi2 = fract(dither * 1.414 + float(i) * 0.381966);

        vec3 sampleDir = CosWeightedHemisphereDir(normalM, xi1, xi2);
        vec3 viewDir = mat3(gbufferModelView) * sampleDir;
        float viewDirLen = length(viewDir);
        if (viewDirLen < 0.001) continue;
        viewDir /= viewDirLen;

        vec3 rayStep = viewDir * (radius * invSteps);
        vec3 rayPos = viewPos + rayStep;

        for (int j = 1; j <= steps; j++) {
            vec4 projPos = gbufferProjection * vec4(rayPos, 1.0);
            vec3 screenPos = projPos.xyz / projPos.w * 0.5 + 0.5;

            if (clamp(screenPos.xy, vec2(0.0), vec2(1.0)) != screenPos.xy) break;

            float sampleDepth = texelFetch(depthtex0, ivec2(screenPos.xy * vec2(viewWidth, viewHeight)), 0).r;
            float sampleLinear = (2.0 * near) / (far + near - sampleDepth * farMinusNear);
            float rayLinear = (2.0 * near) / (far + near - screenPos.z * farMinusNear);

            float depthDiff = rayLinear - sampleLinear;

            if (depthDiff > 0.0 && depthDiff < thicknessThreshold) {
                float dist = float(j) * invSteps;
                float falloff = 1.0 - dist * dist;
                occlusion += falloff;
                totalWeight += 1.0;
                break;
            }

            rayPos += rayStep;
        }
        totalWeight += 1.0;
    }

    if (totalWeight > 0.0) {
        float ao = 1.0 - (occlusion / totalWeight);
        float strength = RT_AO_STRENGTH * 0.01;
        return mix(1.0, ao, strength);
    }

    return 1.0;
}
#endif
