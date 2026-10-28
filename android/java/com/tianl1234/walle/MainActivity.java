package com.tianl1234.walle;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;

/**
 * Wall-E console: runs the native UCI engine, shows its output, and offers
 * presets for the engine's test commands. Everything the engine prints (both
 * stdout and stderr) lands in the transcript, including the "NNUE loaded" line
 * that proves the network was found.
 */
public final class MainActivity extends Activity implements UciEngine.Listener {

    private static final String ENGINE_LIBRARY = "libengine.so";
    private static final String NETWORK_ASSET = "nnue/fingolfin.nnue";
    private static final int MAX_TRANSCRIPT_CHARS = 200000;

    private final StringBuilder transcript = new StringBuilder();

    private TextView logView;
    private TextView statusView;
    private EditText fenInput;
    private EditText commandInput;
    private UciEngine engine;
    private File networkFile;
    /** Bumped on every start so a dying engine cannot overwrite the new one's status. */
    private int engineGeneration;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildLayout());

        networkFile = new File(getFilesDir(), "nnue/fingolfin.nnue");
        append("Wall-E console ready.");
        prepareNetwork();
    }

    // ---------------------------------------------------------------- UI ----

    private View buildLayout() {
        final int pad = dp(12);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        root.setBackgroundColor(Color.parseColor("#121317"));

        statusView = new TextView(this);
        statusView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        statusView.setTextColor(Color.parseColor("#8FA3BF"));
        statusView.setText("engine: starting");
        root.addView(statusView, matchWrap());

        logView = new TextView(this);
        logView.setTypeface(Typeface.MONOSPACE);
        logView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        logView.setTextColor(Color.parseColor("#E6E6E6"));
        logView.setMovementMethod(new ScrollingMovementMethod());
        logView.setTextIsSelectable(true);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(logView, matchWrap());
        LinearLayout.LayoutParams scrollParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f);
        scrollParams.topMargin = dp(8);
        root.addView(scroll, scrollParams);

        fenInput = new EditText(this);
        fenInput.setHint("FEN, empty = startpos");
        fenInput.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        fenInput.setSingleLine(true);
        fenInput.setHintTextColor(Color.parseColor("#5B6472"));
        fenInput.setTextColor(Color.parseColor("#E6E6E6"));
        root.addView(fenInput, matchWrap());

        LinearLayout expectedRow = new LinearLayout(this);
        expectedRow.setOrientation(LinearLayout.HORIZONTAL);
        expectedRow.setGravity(Gravity.CENTER_VERTICAL);
        root.addView(expectedRow, matchWrap());

        commandInput = new EditText(this);
        commandInput.setHint("UCI command, e.g. go depth 16");
        commandInput.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        commandInput.setSingleLine(true);
        commandInput.setHintTextColor(Color.parseColor("#5B6472"));
        commandInput.setTextColor(Color.parseColor("#E6E6E6"));
        commandInput.setLayoutParams(new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        expectedRow.addView(commandInput);

        addButton(expectedRow, "Send", new Runnable() {
            @Override public void run() {
                String command = commandInput.getText().toString().trim();
                if (!command.isEmpty()) {
                    send(command);
                    commandInput.setText("");
                }
            }
        });

        LinearLayout presets = new LinearLayout(this);
        presets.setOrientation(LinearLayout.HORIZONTAL);
        presets.setPadding(0, dp(6), 0, 0);
        root.addView(presets, matchWrap());

        addButton(presets, "Bench", new Runnable() {
            @Override public void run() { send("bench 12"); }
        });
        addButton(presets, "Perft 5", new Runnable() {
            @Override public void run() {
                send(positionCommand());
                send("perft 5");
            }
        });
        addButton(presets, "Analyze", new Runnable() {
            @Override public void run() {
                send(positionCommand());
                send("go depth 16");
            }
        });

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.HORIZONTAL);
        controls.setPadding(0, dp(6), 0, 0);
        root.addView(controls, matchWrap());

        addButton(controls, "Stop", new Runnable() {
            @Override public void run() { send("stop"); }
        });
        addButton(controls, "Restart", new Runnable() {
            @Override public void run() { restartEngine(); }
        });
        addButton(controls, "Clear", new Runnable() {
            @Override public void run() {
                transcript.setLength(0);
                logView.setText("");
            }
        });

        return root;
    }

    private Button addButton(LinearLayout parent, String label, final Runnable action) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        button.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { action.run(); }
        });
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        parent.addView(button, params);
        return button;
    }

    private static LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    // ------------------------------------------------------------ engine ----

    private void prepareNetwork() {
        statusView.setText("network: extracting");
        new Thread(new Runnable() {
            @Override public void run() {
                String error = null;
                try {
                    copyNetworkIfNeeded();
                } catch (IOException e) {
                    error = e.getMessage();
                }
                final String failure = error;
                runOnUiThread(new Runnable() {
                    @Override public void run() {
                        if (failure != null) {
                            append("network error: " + failure);
                            statusView.setText("network: missing");
                        } else {
                            append(String.format(Locale.US, "network ready: %s (%d bytes)",
                                    networkFile.getAbsolutePath(), networkFile.length()));
                            startEngine();
                        }
                    }
                });
            }
        }, "network-setup").start();
    }

    private void copyNetworkIfNeeded() throws IOException {
        File parent = networkFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("cannot create " + parent);
        }
        try (InputStream asset = getAssets().open(NETWORK_ASSET)) {
            int assetSize = asset.available();
            if (networkFile.isFile() && networkFile.length() == assetSize) {
                return;   // already extracted by a previous launch
            }
            try (OutputStream out = new FileOutputStream(networkFile)) {
                byte[] buffer = new byte[1 << 16];
                int read;
                while ((read = asset.read(buffer)) > 0) {
                    out.write(buffer, 0, read);
                }
            }
        }
    }

    private void startEngine() {
        File binary = new File(getApplicationInfo().nativeLibraryDir, ENGINE_LIBRARY);
        if (!binary.isFile()) {
            append("engine binary missing: " + binary);
            statusView.setText("engine: missing binary");
            return;
        }
        try {
            final int generation = ++engineGeneration;
            engine = new UciEngine(binary, getFilesDir(), new UciEngine.Listener() {
                @Override public void onEngineLine(String line, boolean isError) {
                    if (generation == engineGeneration) MainActivity.this.onEngineLine(line, isError);
                }

                @Override public void onEngineExit(int exitCode) {
                    if (generation == engineGeneration) MainActivity.this.onEngineExit(exitCode);
                }
            });
            engine.send("uci");
            engine.send("setoption name EvalFile value " + networkFile.getAbsolutePath());
            engine.send("isready");
            statusView.setText("engine: running (" + binary.getAbsolutePath() + ")");
        } catch (IOException e) {
            append("failed to start engine: " + e.getMessage());
            statusView.setText("engine: failed");
        }
    }

    private void restartEngine() {
        if (engine != null) {
            engine.close();
            engine = null;
        }
        append("--- restart ---");
        startEngine();
    }

    private void send(String command) {
        if (engine == null) {
            append("engine is not running");
            return;
        }
        append("> " + command);
        engine.send(command);
    }

    private String positionCommand() {
        String fen = fenInput.getText().toString().trim();
        return fen.isEmpty() ? "position startpos" : "position fen " + fen;
    }

    private void append(String line) {
        transcript.append(line).append('\n');
        if (transcript.length() > MAX_TRANSCRIPT_CHARS) {
            transcript.delete(0, transcript.length() - MAX_TRANSCRIPT_CHARS);
        }
        logView.setText(transcript);
        logView.post(new Runnable() {
            @Override public void run() {
                ScrollView parent = (ScrollView) logView.getParent();
                if (parent != null) parent.fullScroll(View.FOCUS_DOWN);
            }
        });
    }

    // ----------------------------------------------------------- callbacks --

    @Override
    public void onEngineLine(final String line, final boolean isError) {
        runOnUiThread(new Runnable() {
            @Override public void run() {
                append(isError ? "! " + line : line);
            }
        });
    }

    @Override
    public void onEngineExit(final int exitCode) {
        runOnUiThread(new Runnable() {
            @Override public void run() {
                append("engine exited with code " + exitCode);
                statusView.setText("engine: stopped");
            }
        });
    }

    @Override
    protected void onDestroy() {
        if (engine != null) {
            engine.close();
            engine = null;
        }
        super.onDestroy();
    }
}
