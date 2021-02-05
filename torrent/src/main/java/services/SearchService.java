package services;

import com.google.protobuf.InvalidProtocolBufferException;
import configuration.ErrorMessages;
import configuration.TorrentException;
import entities.Torrent;
import repositories.TorrentRepository;
import utils.Utils;

import java.io.IOException;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

class SearchService {
    private TorrentRepository torrentRepository;

    SearchService(TorrentRepository torrentRepository) {
        this.torrentRepository = torrentRepository;
    }

    Torrent.Message handleSearchRequest(Torrent.SearchRequest searchRequest) {
        Torrent.SearchResponse searchResponse;
        if (!Utils.validateRegex(searchRequest.getRegex())) {
            searchResponse = Torrent.SearchResponse.newBuilder()
                    .setStatus(Torrent.Status.MESSAGE_ERROR)
                    .setErrorMessage(ErrorMessages.INVALID_REGEX)
                    .build();
        } else {
            try {
                Torrent.SubnetResponse subnetResponse = SubnetService.getSubnets(searchRequest.getSubnetId());
                List<Torrent.NodeSearchResult> nodeSearchResults = this.getSearchResults(searchRequest.getRegex(), subnetResponse.getNodesList());
                searchResponse = Torrent.SearchResponse.newBuilder()
                        .addAllResults(nodeSearchResults)
                        .setStatus(Torrent.Status.SUCCESS)
                        .build();
            } catch (TorrentException e) {
                searchResponse = Torrent.SearchResponse.newBuilder()
                        .setStatus(e.getStatus())
                        .setErrorMessage(e.getMessage())
                        .build();
            }
        }

        return Torrent.Message.newBuilder()
                .setType(Torrent.Message.Type.SEARCH_RESPONSE)
                .setSearchResponse(searchResponse)
                .build();
    }

    private List<Torrent.NodeSearchResult> getSearchResults(String regex, List<Torrent.NodeId> nodeIds) {
        List<Torrent.NodeSearchResult> results = new ArrayList<>();
        List<Callable<Torrent.NodeSearchResult>> tasks = new ArrayList<>();
        int threadSize = nodeIds.size();
        if (nodeIds.contains(torrentRepository.getNodeId())) {
            threadSize--;
        }
        ExecutorService executorService = Executors.newFixedThreadPool(threadSize);
        for (Torrent.NodeId nodeId : nodeIds) {
            if (!nodeId.equals(torrentRepository.getNodeId())) {
                tasks.add(new SearchThread(regex, nodeId));
            } else {
                results.add(Torrent.NodeSearchResult.newBuilder()
                        .setNode(nodeId)
                        .setStatus(Torrent.Status.SUCCESS)
                        .addAllFiles(torrentRepository.getFileInfoByFileNameRegex(regex))
                        .build());
            }
        }
        try {
            List<Future<Torrent.NodeSearchResult>> futures = executorService.invokeAll(tasks);
            for (Future<Torrent.NodeSearchResult> future : futures) {
                Torrent.NodeSearchResult nodeSearchResult = future.get();
                results.add(nodeSearchResult);
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
        return results;
    }


    public class SearchThread implements Callable<Torrent.NodeSearchResult> {
        private String regex;
        private Torrent.NodeId nodeId;

        SearchThread(String regex, Torrent.NodeId nodeId) {
            super();
            this.regex = regex;
            this.nodeId = nodeId;
        }

        @Override
        public Torrent.NodeSearchResult call() {
            Torrent.NodeSearchResult nodeSearchResult;
            try {
                Socket socket = new Socket(nodeId.getHost(), nodeId.getPort());
                Torrent.LocalSearchRequest localSearchRequest = Torrent.LocalSearchRequest.newBuilder().setRegex(regex).build();
                Torrent.Message requestMessage = Torrent.Message.newBuilder()
                        .setType(Torrent.Message.Type.LOCAL_SEARCH_REQUEST)
                        .setLocalSearchRequest(localSearchRequest)
                        .build();
                Utils.writeMessage(requestMessage, socket.getOutputStream());
                Torrent.Message responseMessage = Utils.readMessage(socket.getInputStream());
                if (responseMessage.getType() != Torrent.Message.Type.LOCAL_SEARCH_RESPONSE) {
                    nodeSearchResult = Torrent.NodeSearchResult.newBuilder()
                            .setStatus(Torrent.Status.MESSAGE_ERROR)
                            .setErrorMessage(ErrorMessages.WRONG_TYPE + " " + responseMessage.getType())
                            .build();
                } else {
                    Torrent.LocalSearchResponse localSearchResponse = responseMessage.getLocalSearchResponse();
                    nodeSearchResult = Torrent.NodeSearchResult.newBuilder()
                            .setStatus(localSearchResponse.getStatus())
                            .setErrorMessage(localSearchResponse.getErrorMessage())
                            .setNode(nodeId)
                            .addAllFiles(localSearchResponse.getFileInfoList())
                            .build();
                }
                socket.close();
            } catch (InvalidProtocolBufferException e) {
                nodeSearchResult = Torrent.NodeSearchResult.newBuilder()
                        .setStatus(Torrent.Status.MESSAGE_ERROR)
                        .setErrorMessage(ErrorMessages.INVALID_FORMAT)
                        .build();
            } catch (IOException e) {
                nodeSearchResult = Torrent.NodeSearchResult.newBuilder()
                        .setStatus(Torrent.Status.NETWORK_ERROR)
                        .setErrorMessage(ErrorMessages.NETWORK_ERROR)
                        .build();
                e.printStackTrace();
            }
            return nodeSearchResult;
        }
    }
}
