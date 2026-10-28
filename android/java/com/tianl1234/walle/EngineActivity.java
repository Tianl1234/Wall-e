package com.tianl1234.walle;

import android.app.Activity;
import android.os.Bundle;

/**
 * Marker activity for the Open Exchange (OEX) engine convention.
 *
 * A chess GUI finds installed engines by querying for the action
 * {@code intent.chess.provider.ENGINE}; the manifest declares it on this activity and
 * points at {@link EngineProvider} through meta-data. Nothing is ever displayed, and
 * there is no launcher entry, so the package has no icon and no UI.
 */
public class EngineActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        finish();
    }
}
