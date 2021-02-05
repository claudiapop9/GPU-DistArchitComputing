package entities;

import com.google.protobuf.ByteString;
import configuration.Constants;

import java.util.HashMap;

public class FileData {
    private HashMap<Integer, ByteString> data;
    private Torrent.FileInfo fileInfo;

    public FileData(ByteString data, Torrent.FileInfo fileInfo) {
        this.data = new HashMap<>();
        for (int i = 0; i < fileInfo.getChunksCount() - 1; i++) {
            this.data.put(i, data.substring(i * Constants.CHUNK_SIZE, (i + 1) * Constants.CHUNK_SIZE ));
        }
        this.data.put(fileInfo.getChunksCount() - 1, data.substring((fileInfo.getChunksCount() - 1) * Constants.CHUNK_SIZE));

        this.fileInfo = fileInfo;
    }

    public FileData(Torrent.FileInfo fileInfo) {
        this.data = new HashMap<>();
        this.fileInfo = fileInfo;
    }

    public HashMap<Integer, ByteString> getData() {
        return data;
    }

    public ByteString getChunk(int chunkIndex) {
        return this.data.get(chunkIndex);
    }

    public void putChunk(int chunkIndex, ByteString chunk) {
        this.data.put(chunkIndex, chunk);
    }


    public Torrent.FileInfo getFileInfo() {
        return fileInfo;
    }

    public void setFileInfo(Torrent.FileInfo fileInfo) {
        this.fileInfo = fileInfo;
    }

}
