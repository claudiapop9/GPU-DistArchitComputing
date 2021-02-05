package services;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import configuration.ErrorMessages;
import configuration.TorrentException;
import entities.FileData;
import entities.NodeReplicationWrapper;
import entities.Torrent;
import repositories.TorrentRepository;
import utils.Utils;

import java.io.IOException;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class ReplicateService {
    private TorrentRepository torrentRepository;

    ReplicateService(TorrentRepository torrentRepository) {
        this.torrentRepository = torrentRepository;
    }

    Torrent.Message handleReplicateRequest(Torrent.ReplicateRequest replicateRequest) {

        if (replicateRequest.getFileInfo().getFilename().length() == 0) {
            return Torrent.Message.newBuilder().setType(Torrent.Message.Type.REPLICATE_RESPONSE)
                    .setUploadResponse(Torrent.UploadResponse.newBuilder()
                            .setStatus(Torrent.Status.MESSAGE_ERROR)
                            .setErrorMessage(ErrorMessages.EMPTY_FILENAME)).build();
        }
        List<Torrent.NodeReplicationStatus> nodeReplicationStatuses = new ArrayList<>();
        if (torrentRepository.getFile(replicateRequest.getFileInfo().getFilename()) == null) {
            try {
                Torrent.SubnetResponse subnetResponse = SubnetService.getSubnets(replicateRequest.getSubnetId());
                nodeReplicationStatuses = this.getChunks(replicateRequest, subnetResponse.getNodesList());
            } catch (TorrentException e) {
                return Torrent.Message.newBuilder()
                        .setType(Torrent.Message.Type.REPLICATE_RESPONSE)
                        .setReplicateResponse(Torrent.ReplicateResponse.newBuilder()
                                .setStatus(e.getStatus())
                                .setErrorMessage(e.getMessage())
                                .build())
                        .build();
            }
        }

        Torrent.ReplicateResponse replicateResponse = Torrent.ReplicateResponse.newBuilder()
                .setStatus(Torrent.Status.SUCCESS)
                .addAllNodeStatusList(nodeReplicationStatuses)
                .build();
        return Torrent.Message.newBuilder()
                .setType(Torrent.Message.Type.REPLICATE_RESPONSE)
                .setReplicateResponse(replicateResponse)
                .build();
    }


    public List<Torrent.NodeReplicationStatus> getChunks(Torrent.ReplicateRequest replicateRequest, List<Torrent.NodeId> nodes) {
        List<Torrent.NodeReplicationStatus> nodeReplicationStatuses = new ArrayList<>();
        ExecutorService executorService = Executors.newFixedThreadPool(replicateRequest.getFileInfo().getChunksCount());
        List<Callable<List<NodeReplicationWrapper>>> tasks = new ArrayList<>();
        for (int i = 0; i < replicateRequest.getFileInfo().getChunksCount(); i++) {
            tasks.add(new ChunkThread(i, replicateRequest.getFileInfo().getHash(), nodes));
        }
        try {
            List<Future<List<NodeReplicationWrapper>>> futures = executorService.invokeAll(tasks);
            FileData fileData = new FileData(replicateRequest.getFileInfo());

            for (Future<List<NodeReplicationWrapper>> future : futures) {
                List<NodeReplicationWrapper> nodeReplicationWrappers = future.get();
                for (NodeReplicationWrapper wrapper : nodeReplicationWrappers) {
                    Torrent.NodeReplicationStatus nodeReplicationStatus = wrapper.getNodeReplicationStatus();
                    nodeReplicationStatuses.add(nodeReplicationStatus);
                    if (nodeReplicationStatus.getStatus() == Torrent.Status.SUCCESS) {
                        fileData.putChunk(nodeReplicationStatus.getChunkIndex(), wrapper.getChunk());
                    }
                }
            }
            executorService.shutdown();
            torrentRepository.putFile(replicateRequest.getFileInfo().getFilename(), fileData);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        return nodeReplicationStatuses;
    }

    public class ChunkThread implements Callable<List<NodeReplicationWrapper>> {
        private int chunkIndex;
        private ByteString hash;
        private List<Torrent.NodeId> nodeIds;

        ChunkThread(int index, ByteString hash, List<Torrent.NodeId> nodes) {
            super();
            this.chunkIndex = index;
            this.hash = hash;
            this.nodeIds = nodes;
        }

        @Override
        public List<NodeReplicationWrapper> call() {
            List<NodeReplicationWrapper> nodeReplicationStatuses = new ArrayList<>();
            int index;
            boolean found = false;
            for (int i = 0; i < nodeIds.size() && !found; i++) {
                index = (chunkIndex + i) % nodeIds.size();
                Torrent.NodeId nodeId = nodeIds.get(index);
                if (!nodeId.equals(torrentRepository.getNodeId())) {
                    try {
                        Socket socket = new Socket(nodeId.getHost(), nodeId.getPort());
                        Torrent.ChunkRequest chunkRequest = Torrent.ChunkRequest.newBuilder().setChunkIndex(this.chunkIndex)
                                .setFileHash(this.hash).build();
                        Torrent.Message requestMessage = Torrent.Message.newBuilder()
                                .setType(Torrent.Message.Type.CHUNK_REQUEST)
                                .setChunkRequest(chunkRequest)
                                .build();
                        Utils.writeMessage(requestMessage, socket.getOutputStream());
                        Torrent.Message responseMessage = Utils.readMessage(socket.getInputStream());
                        if (responseMessage.getType() != Torrent.Message.Type.CHUNK_RESPONSE) {
                            nodeReplicationStatuses.add(new NodeReplicationWrapper(Torrent.NodeReplicationStatus.newBuilder()
                                    .setNode(nodeId)
                                    .setChunkIndex(chunkIndex)
                                    .setStatus(Torrent.Status.MESSAGE_ERROR)
                                    .setErrorMessage(ErrorMessages.INVALID_FORMAT)
                                    .build()));
                        } else {
                            Torrent.ChunkResponse chunkResponse = responseMessage.getChunkResponse();
                            ByteString chunk = null;
                            if (chunkResponse.getStatus() == Torrent.Status.SUCCESS) {
                                found = true;
                                chunk = chunkResponse.getData();
                            }
                            nodeReplicationStatuses.add(new NodeReplicationWrapper(Torrent.NodeReplicationStatus.newBuilder()
                                    .setNode(nodeId)
                                    .setChunkIndex(chunkIndex)
                                    .setStatus(chunkResponse.getStatus())
                                    .setErrorMessage(chunkResponse.getErrorMessage())
                                    .build(), chunk));
                        }
                        socket.close();
                    } catch (InvalidProtocolBufferException e) {
                        nodeReplicationStatuses.add(new NodeReplicationWrapper(Torrent.NodeReplicationStatus.newBuilder()
                                .setNode(nodeId)
                                .setChunkIndex(chunkIndex)
                                .setStatus(Torrent.Status.MESSAGE_ERROR)
                                .setErrorMessage(ErrorMessages.INVALID_FORMAT)
                                .build()));

                    } catch (IOException e) {
                        e.printStackTrace();
                        nodeReplicationStatuses.add(new NodeReplicationWrapper(Torrent.NodeReplicationStatus.newBuilder()
                                .setNode(nodeId)
                                .setChunkIndex(chunkIndex)
                                .setStatus(Torrent.Status.NETWORK_ERROR)
                                .setErrorMessage(ErrorMessages.NETWORK_ERROR)
                                .build()));
                    }
                }
            }
            return nodeReplicationStatuses;
        }
    }
}
