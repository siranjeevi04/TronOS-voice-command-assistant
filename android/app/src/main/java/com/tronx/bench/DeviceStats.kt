package com.tronx.bench

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES20
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.io.File

/** Process memory, thermal, CPU-frequency, battery and device-info readers (all real Android APIs / procfs). */
object DeviceStats {

    /** VmRSS / VmHWM of THIS process in KB, from /proc/self/status. */
    fun procStatusKb(key: String): Long {
        return try {
            File("/proc/self/status").readLines().firstOrNull { it.startsWith("$key:") }
                ?.replace(Regex("[^0-9]"), "")?.toLong() ?: -1L
        } catch (e: Exception) { -1L }
    }

    fun rssMb(): Double = procStatusKb("VmRSS") / 1024.0
    fun hwmMb(): Double = procStatusKb("VmHWM") / 1024.0

    /** Background sampler: tracks max VmRSS every [periodMs] (captures peaks that a single read would miss). */
    class PeakSampler(private val periodMs: Long = 15) {
        @Volatile private var running = false
        @Volatile var peakMb = 0.0
            private set
        private var thread: Thread? = null
        fun start() {
            peakMb = rssMb(); running = true
            thread = Thread {
                while (running) {
                    val r = rssMb(); if (r > peakMb) peakMb = r
                    try { Thread.sleep(periodMs) } catch (_: InterruptedException) { }
                }
            }.also { it.isDaemon = true; it.start() }
        }
        fun stop(): Double { running = false; thread?.join(200); val r = rssMb(); if (r > peakMb) peakMb = r; return peakMb }
    }

    fun cpuFreqsMhz(): List<String> {
        val n = Runtime.getRuntime().availableProcessors()
        return (0 until n).map { i ->
            try { (File("/sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq").readText().trim().toLong() / 1000).toString() }
            catch (e: Exception) { "n/a" }
        }
    }

    fun cpuMaxFreqsMhz(): List<String> {
        val n = Runtime.getRuntime().availableProcessors()
        return (0 until n).map { i ->
            try { (File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim().toLong() / 1000).toString() }
            catch (e: Exception) { "n/a" }
        }
    }

    /** Readable thermal zones (many are permission-restricted on production phones; unreadable ones are skipped). */
    fun thermalZones(): List<Pair<String, Double>> {
        val out = mutableListOf<Pair<String, Double>>()
        val dir = File("/sys/class/thermal")
        dir.listFiles { f -> f.name.startsWith("thermal_zone") }?.sortedBy { it.name }?.forEach { z ->
            try {
                val type = File(z, "type").readText().trim()
                val t = File(z, "temp").readText().trim().toDouble()
                val c = if (t > 1000) t / 1000.0 else t
                if (c > 0 && c < 150) out.add(type to c)
            } catch (_: Exception) { }
        }
        return out
    }

    data class Battery(val levelPct: Int, val tempC: Double, val voltageMv: Int, val plugged: Boolean, val currentUa: Long, val chargeUah: Long)

    fun battery(ctx: Context): Battery {
        val i = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = i?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = i?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val temp = (i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -10) ?: -10) / 10.0
        val volt = i?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
        val plugged = (i?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val cur = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val chg = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        return Battery(if (scale > 0) level * 100 / scale else level, temp, volt, plugged, cur, chg)
    }

    fun thermalStatus(ctx: Context): String {
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        return when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "NONE(0)"
            PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT(1)"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE(2)"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE(3)"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL(4)"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY(5)"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN(6)"
            else -> "UNKNOWN"
        }
    }

    fun thermalStatusInt(ctx: Context): Int = (ctx.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus

    /** Thermal headroom (API 30+): 1.0 = throttling threshold reached; NaN if unsupported. */
    fun thermalHeadroom(ctx: Context): Float {
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        return if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(0) else Float.NaN
    }

    fun powerSave(ctx: Context): Boolean = (ctx.getSystemService(Context.POWER_SERVICE) as PowerManager).isPowerSaveMode

    fun totalRamMb(ctx: Context): Long {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo(); am.getMemoryInfo(mi); return mi.totalMem / (1024 * 1024)
    }

    fun availRamMb(ctx: Context): Long {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo(); am.getMemoryInfo(mi); return mi.availMem / (1024 * 1024)
    }

    /** GPU renderer string via a tiny off-screen EGL/GLES2 context. */
    fun gpuName(): String {
        return try {
            val dpy = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val ver = IntArray(2); EGL14.eglInitialize(dpy, ver, 0, ver, 1)
            val attribs = intArrayOf(EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT, EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, EGL14.EGL_NONE)
            val cfgs = arrayOfNulls<EGLConfig>(1); val num = IntArray(1)
            EGL14.eglChooseConfig(dpy, attribs, 0, cfgs, 0, 1, num, 0)
            val ctx = EGL14.eglCreateContext(dpy, cfgs[0], EGL14.EGL_NO_CONTEXT, intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE), 0)
            val surf = EGL14.eglCreatePbufferSurface(dpy, cfgs[0], intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
            EGL14.eglMakeCurrent(dpy, surf, surf, ctx)
            val r = "${GLES20.glGetString(GLES20.GL_VENDOR)} ${GLES20.glGetString(GLES20.GL_RENDERER)} (${GLES20.glGetString(GLES20.GL_VERSION)})"
            EGL14.eglMakeCurrent(dpy, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            EGL14.eglDestroySurface(dpy, surf); EGL14.eglDestroyContext(dpy, ctx); EGL14.eglTerminate(dpy)
            r
        } catch (e: Exception) { "unavailable (${e.message})" }
    }

    fun socName(): String = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE

    fun cpuDescription(): String {
        val maxes = cpuMaxFreqsMhz()
        val groups = maxes.groupingBy { it }.eachCount().entries.joinToString(" + ") { "${it.value}x ${it.key} MHz" }
        val impl = try {
            File("/proc/cpuinfo").readLines().filter { it.startsWith("CPU part") }.map { it.substringAfter(":").trim() }
                .groupingBy { it }.eachCount().entries.joinToString(", ") { "${it.value}x part ${it.key}" }
        } catch (e: Exception) { "" }
        return "${Runtime.getRuntime().availableProcessors()} cores: $groups ${if (impl.isNotEmpty()) "[$impl]" else ""}"
    }
}
