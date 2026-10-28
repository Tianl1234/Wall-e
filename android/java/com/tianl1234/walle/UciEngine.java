package com.tianl1234.walle;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;

/**
 * Runs the native engine as a child process and speaks UCI over its stdio.
 *
 * The binary is shipped as libengine.so inside the APK: Android extracts native
 * libraries into an executable directory, while files written to the app's data
 * directory cannot be executed at all on Android 10+ (W^X).
 */
public final class UciEngine {

    public interface Listener {
        void onEngineLine(String line, boolean isError);
        void onEngineExit(int exitCode);
    }

    private final Process process;
    private final BufferedWriter stdin;

    public UciEngine(File binary, File workingDirectory, Listener listener) throws IOException {
        ProcessBuilder builder = new ProcessBuilder(binary.getAbsolutePath());
        builder.directory(workingDirectory);
        builder.redirectErrorStream(false);
        process = builder.start();

        stdin = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
        pump(process.getInputStream(), listener, false);
        pump(process.getErrorStream(), listener, true);

        Thread waiter = new Thread(new Runnable() {
            @Override public void run() {
                try {
                    final int code = process.waitFor();
                    listener.onEngineExit(code);
                } catch (InterruptedException ignored) {
                    // Activity is going away; nothing to report.
                }
            }
        }, "engine-exit");
        waiter.setDaemon(true);
        waiter.start();
    }

    private static void pump(final InputStream stream, final Listener listener, final boolean isError) {
        Thread thread = new Thread(new Runnable() {
            @Override public void run() {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(stream, StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        listener.onEngineLine(line, isError);
                    }
                } catch (IOException ignored) {
                    // Engine closed its pipe.
                }
            }
        }, isError ? "engine-stderr" : "engine-stdout");
        thread.setDaemon(true);
        thread.start();
    }

    public synchronized void send(String command) {
        try {
            stdin.write(command);
            stdin.write('\n');
            stdin.flush();
        } catch (IOException ignored) {
            // Broken pipe: the exit callback already reports the failure.
        }
    }

    public synchronized void close() {
        send("quit");
        process.destroy();
    }
}
