package services;

import com.google.protobuf.ByteString;
import configuration.ErrorMessages;
import entities.Torrent;
import repositories.TorrentRepository;
import utils.Utils;

public class ChunkService {
    private TorrentRepository torrentRepository;

    public ChunkService(TorrentRepository torrentRepository) {
        this.torrentRepository = torrentRepository;
    }

    Torrent.Message handleChunkRequest(Torrent.ChunkRequest chunkRequest) {
        Torrent.ChunkResponse chunkResponse;
        if (chunkRequest.getChunkIndex() < 0 || !Utils.validateMd5(chunkRequest.getFileHash())) {
            String message = chunkRequest.getChunkIndex() < 0 ? ErrorMessages.INVALID_INDEX : ErrorMessages.INVALID_HASH;
            chunkResponse = Torrent.ChunkResponse.newBuilder()
                    .setErrorMessage(message)
                    .setStatus(Torrent.Status.MESSAGE_ERROR)
                    .build();
        } else {
            ByteString chunk = torrentRepository.getChunkByHash(chunkRequest.getFileHash(), chunkRequest.getChunkIndex());
            if (chunk == null) {
                chunkResponse = Torrent.ChunkResponse.newBuilder()
                        .setErrorMessage(ErrorMessages.FILE_NOT_FOUND)
                        .setStatus(Torrent.Status.UNABLE_TO_COMPLETE)
                        .build();
            } else {
                chunkResponse = Torrent.ChunkResponse.newBuilder()
                        .setStatus(Torrent.Status.SUCCESS)
                        .setData(chunk)
                        .build();
            }
        }
        return Torrent.Message.newBuilder().setType(Torrent.Message.Type.CHUNK_RESPONSE).setChunkResponse(chunkResponse).build();
    }
}