package repositories;

import com.google.protobuf.ByteString;
import entities.FileData;
import entities.Torrent;

import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class TorrentRepository {
    private HashMap<String, FileData> files;
    private Torrent.NodeId nodeId;

    public TorrentRepository(Torrent.NodeId nodeId) {
        this.files = new HashMap<String, FileData>();
        this.nodeId = nodeId;
    }

    public HashMap<String, FileData> getFiles() {
        return this.files;
    }

    public FileData getFile(String fileName) {
        return this.files.get(fileName);
    }

    public Torrent.NodeId getNodeId() {
        return this.nodeId;
    }

    public void putFile(String key, FileData fileData) {
        this.files.put(key, fileData);
    }

    public FileData findByHash(ByteString hash) {
        return this.files.values().stream().filter(fileData -> fileData.getFileInfo().getHash().equals(hash)).findFirst().orElse(null);
    }

    public ByteString getChunkByHash(ByteString hash, int chunkIndex) {
        FileData fileData = this.findByHash(hash);
        if (fileData != null) {
            if (chunkIndex >= fileData.getFileInfo().getChunksCount()) {
                return null;
            }
            return fileData.getChunk(chunkIndex);

        }
        return null;
    }

    public ByteString getDataByHash(ByteString hash) {
        FileData fileData = this.findByHash(hash);
        if (fileData != null) {
            ByteString byteString = fileData.getData().get(0);
            for (int i = 1; i < fileData.getFileInfo().getChunksCount(); i++) {
                byteString = byteString.concat(fileData.getData().get(i));
            }
            return byteString;
        }
        return null;
    }

    public List<Torrent.FileInfo> getFileInfoByFileNameRegex(String regex) {
        return this.files.values().stream().map(fileData -> fileData.getFileInfo()).filter(fileInfo -> fileInfo.getFilename().matches(regex)).collect(Collectors.toList());
    }
}
