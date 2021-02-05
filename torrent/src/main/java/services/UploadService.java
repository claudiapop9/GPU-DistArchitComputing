package services;

import com.google.protobuf.ByteString;
import configuration.Constants;
import configuration.ErrorMessages;
import entities.FileData;
import entities.Torrent;
import repositories.TorrentRepository;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;

class UploadService {
    private TorrentRepository torrentRepository;

    UploadService(TorrentRepository torrentRepository) {
        this.torrentRepository = torrentRepository;
    }

    Torrent.Message handleUploadRequest(Torrent.UploadRequest uploadRequest) {
        System.out.println(uploadRequest);
        if (uploadRequest.getFilename().length() == 0) {
            return Torrent.Message.newBuilder().setType(Torrent.Message.Type.UPLOAD_RESPONSE)
                    .setUploadResponse(Torrent.UploadResponse.newBuilder()
                            .setStatus(Torrent.Status.MESSAGE_ERROR)
                            .setErrorMessage(ErrorMessages.EMPTY_FILENAME)).build();
        }
        if (this.torrentRepository.getFile(uploadRequest.getFilename()) == null) {
            FileData fileData = this.getFileDataFromRequest(uploadRequest);
            torrentRepository.putFile(fileData.getFileInfo().getFilename(), fileData);
        } else {
            System.out.println(ErrorMessages.UPLOADED_FILE + uploadRequest.getFilename());
        }
        Torrent.UploadResponse uploadResponse = Torrent.UploadResponse.newBuilder()
                .setStatus(Torrent.Status.SUCCESS)
                .setFileInfo(this.torrentRepository.getFile(uploadRequest.getFilename()).getFileInfo()).build();
        return Torrent.Message.newBuilder()
                .setType(Torrent.Message.Type.UPLOAD_RESPONSE)
                .setUploadResponse(uploadResponse)
                .build();
    }

    private FileData getFileDataFromRequest(Torrent.UploadRequest uploadRequest) {
        ByteString data = uploadRequest.getData();
        Torrent.FileInfo fileInfo;
        if (data.size() == 0) {
            fileInfo = Torrent.FileInfo.newBuilder().setSize(0).setFilename(uploadRequest.getFilename()).build();
        } else {
            int noOfChunks = data.size() / Constants.CHUNK_SIZE;
            if (data.size() % Constants.CHUNK_SIZE != 0) {
                noOfChunks++;
            }
            List<Torrent.ChunkInfo> chunks = new ArrayList<>();
            for (int chunkId = 0; chunkId < noOfChunks - 1; chunkId++) {
                int index = chunkId * Constants.CHUNK_SIZE;
                Torrent.ChunkInfo chunkInfo = Torrent.ChunkInfo.newBuilder()
                        .setIndex(chunkId)
                        .setSize(Constants.CHUNK_SIZE)
                        .setHash(Utils.encryptData(data.substring(index, index + Constants.CHUNK_SIZE)))
                        .build();
                chunks.add(chunkInfo);
            }
            ByteString lastChunkHash = data.substring((noOfChunks - 1) * Constants.CHUNK_SIZE);
            chunks.add(Torrent.ChunkInfo.newBuilder()
                    .setIndex(noOfChunks - 1)
                    .setSize(lastChunkHash.size())
                    .setHash(Utils.encryptData(lastChunkHash))
                    .build());
            fileInfo = Torrent.FileInfo.newBuilder()
                    .setHash(Utils.encryptData(data))
                    .setSize(data.size())
                    .setFilename(uploadRequest.getFilename())
                    .addAllChunks(chunks)
                    .build();
        }
        return new FileData(data, fileInfo);
    }
}
