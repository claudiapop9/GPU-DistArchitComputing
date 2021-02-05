package entities;

import com.google.protobuf.ByteString;

public class NodeReplicationWrapper {
    private Torrent.NodeReplicationStatus nodeReplicationStatus;
    private ByteString chunk;

    public NodeReplicationWrapper(Torrent.NodeReplicationStatus nodeReplicationStatus, ByteString chunk) {
        this.nodeReplicationStatus = nodeReplicationStatus;
        this.chunk = chunk;
    }

    public NodeReplicationWrapper(Torrent.NodeReplicationStatus nodeReplicationStatus) {
        this.nodeReplicationStatus = nodeReplicationStatus;
    }

    public Torrent.NodeReplicationStatus getNodeReplicationStatus() {
        return nodeReplicationStatus;
    }

    public void setNodeReplicationStatus(Torrent.NodeReplicationStatus nodeReplicationStatus) {
        this.nodeReplicationStatus = nodeReplicationStatus;
    }

    public ByteString getChunk() {
        return chunk;
    }

    public void setChunk(ByteString chunk) {
        this.chunk = chunk;
    }
}
