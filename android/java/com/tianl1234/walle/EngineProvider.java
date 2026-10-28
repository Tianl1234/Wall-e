package com.tianl1234.walle;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;

/**
 * OEX engine provider: hands the engine binary to any chess GUI that asks for it.
 *
 * Protocol (matching the established Android engine packages):
 *   content://com.tianl1234.walle.EngineProvider/<file name>
 * Looked up in assets first, then in the app's nativeLibraryDir — which is where
 * {@code lib/arm64-v8a/libwalle.so} and {@code lib/x86_64/libwalle.so} are extracted.
 *
 * Most GUIs do not read this provider at all: the OEX resolver builds the path
 * {@code ApplicationInfo.nativeLibraryDir + "/" + filename} from res/xml/enginelist.xml
 * and executes the file in place. This provider exists for the GUIs that copy the
 * binary through a content resolver instead.
 */
public class EngineProvider extends ContentProvider {
    private static final String TAG = "WallEProvider";
    private static final String MIME_TYPE = "application/x-chess-engine";
    private static final String UNSUPPORTED = "Not supported by this provider";

    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public String getType(Uri uri) {
        return MIME_TYPE;
    }

    @Override
    public AssetFileDescriptor openAssetFile(Uri uri, String mode) throws FileNotFoundException {
        String fileName = uri.getLastPathSegment();
        if (fileName != null) {
            try {
                return getContext().getAssets().openFd(fileName);
            } catch (Exception notAnAsset) {
                Log.d(TAG, "Engine file <" + fileName
                        + "> was not found in assets, trying to load from libraries.");
            }
        }

        String path = getContext().getApplicationInfo().nativeLibraryDir
                + File.separator + fileName;
        try {
            return new AssetFileDescriptor(openLibFile(new File(path)), 0, -1);
        } catch (Exception e) {
            Log.e(TAG, "Error opening file <" + path + ">.", e);
            throw new FileNotFoundException(path);
        }
    }

    public ParcelFileDescriptor openLibFile(File file) throws FileNotFoundException {
        return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY);
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection,
                        String[] selectionArgs, String sortOrder) {
        throw new UnsupportedOperationException(UNSUPPORTED);
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException(UNSUPPORTED);
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException(UNSUPPORTED);
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException(UNSUPPORTED);
    }
}
