package services;

import com.google.protobuf.ByteString;
import configuration.ErrorMessages;
import entities.Torrent;
import repositories.TorrentRepository;
import utils.Utils;

class DownloadService {
    private TorrentRepository torrentRepository;

    DownloadService(TorrentRepository torrentRepository) {
        this.torrentRepository = torrentRepository;
    }

    Torrent.Message handleDownloadRequest(Torrent.DownloadRequest downloadRequest) {
        Torrent.DownloadResponse downloadResponse;
        ByteString fileHash = downloadRequest.getFileHash();
        if (!Utils.validateMd5(fileHash)) {
            downloadResponse = Torrent.DownloadResponse.newBuilder()
                    .setStatus(Torrent.Status.MESSAGE_ERROR)
                    .setErrorMessage(ErrorMessages.INVALID_HASH)
                    .build();
        } else {
            ByteString data = torrentRepository.getDataByHash(downloadRequest.getFileHash());
            if (data == null) {
                downloadResponse = Torrent.DownloadResponse.newBuilder()
                        .setStatus(Torrent.Status.UNABLE_TO_COMPLETE)
                        .setErrorMessage(ErrorMessages.FILE_NOT_FOUND)
                        .build();
            } else {
                downloadResponse = Torrent.DownloadResponse.newBuilder()
                        .setStatus(Torrent.Status.SUCCESS)
                        .setData(data).build();
            }
        }
        return Torrent.Message.newBuilder().setType(Torrent.Message.Type.DOWNLOAD_RESPONSE)
                .setDownloadResponse(downloadResponse).build();
    }
}
