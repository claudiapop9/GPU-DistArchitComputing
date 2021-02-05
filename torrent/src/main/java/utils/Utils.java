package utils;

import com.google.protobuf.ByteString;
import configuration.Constants;
import entities.Torrent;

import java.io.IOException;

import java.io.InputStream;
import java.io.OutputStream;

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

public class Utils {

    public static void writeMessage(Torrent.Message message, OutputStream outputStream) throws IOException {
        byte[] bytes = message.toByteArray();
        outputStream.write(ByteBuffer.allocate(Constants.INT_SIZE).putInt(bytes.length).array());
        outputStream.write(bytes);
    }

    public static Torrent.Message readMessage(InputStream inputStream) throws IOException {
        int length = ByteBuffer.wrap(inputStream.readNBytes(Constants.INT_SIZE)).getInt();
        return Torrent.Message.parseFrom(inputStream.readNBytes(length));
    }

    public static ByteString encryptData(ByteString input) {
        try {
            return ByteString.copyFrom(MessageDigest.getInstance("MD5").digest(input.toByteArray()));
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static boolean validateRegex(String regex) {
        try {
            Pattern.compile(regex);
            return true;
        } catch (PatternSyntaxException e) {
            return false;
        }
    }

    public static boolean validateMd5(ByteString hash) {
        return hash.size() == 16;
    }
}
