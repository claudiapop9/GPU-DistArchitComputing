package services;

import configuration.ErrorMessages;
import entities.Torrent;
import repositories.TorrentRepository;
import utils.Utils;

import java.util.List;

class LocalSearchService {
    private TorrentRepository torrentRepository;

    LocalSearchService(TorrentRepository torrentRepository) {
        this.torrentRepository = torrentRepository;
    }

    Torrent.Message handleLocalSearchRequest(Torrent.LocalSearchRequest localSearchRequest) {
        String regex = localSearchRequest.getRegex();
        if (!Utils.validateRegex(regex)) {
            return Torrent.Message.newBuilder()
                    .setType(Torrent.Message.Type.LOCAL_SEARCH_RESPONSE)
                    .setLocalSearchResponse(Torrent.LocalSearchResponse.newBuilder()
                            .setStatus(Torrent.Status.MESSAGE_ERROR)
                            .setErrorMessage(ErrorMessages.INVALID_REGEX).build())
                    .build();
        }
        Torrent.LocalSearchResponse localSearchResponse;
        List<Torrent.FileInfo> fileInfos = torrentRepository.getFileInfoByFileNameRegex(regex);
        localSearchResponse = Torrent.LocalSearchResponse
                .newBuilder()
                .addAllFileInfo(fileInfos)
                .setStatus(Torrent.Status.SUCCESS)
                .build();
        return Torrent.Message.newBuilder()
                .setType(Torrent.Message.Type.LOCAL_SEARCH_RESPONSE)
                .setLocalSearchResponse(localSearchResponse)
                .build();
    }
}
