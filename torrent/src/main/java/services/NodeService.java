package services;

import configuration.Constants;
import entities.Torrent;
import repositories.TorrentRepository;
import utils.Utils;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;

public class NodeService {
    private boolean registered;
    private TorrentRepository torrentRepository;
    private UploadService uploadService;
    private ReplicateService replicateService;
    private ChunkService chunkService;
    private LocalSearchService localSearchService;
    private SearchService searchService;
    private DownloadService downloadService;

    public NodeService(int port, int index) {
        Torrent.NodeId nodeid = Torrent.NodeId.newBuilder()
                .setIndex(index)
                .setPort(port)
                .setHost(Constants.HOST)
                .setOwner(Constants.OWNER)
                .build();
        this.torrentRepository = new TorrentRepository(nodeid);
        uploadService = new UploadService(torrentRepository);
        replicateService = new ReplicateService(torrentRepository);
        chunkService = new ChunkService(torrentRepository);
        localSearchService = new LocalSearchService(torrentRepository);
        searchService = new SearchService(torrentRepository);
        downloadService = new DownloadService(torrentRepository);
    }

    public boolean isRegistered() {
        return this.registered;
    }

    public Torrent.NodeId getNodeId() {
        return this.torrentRepository.getNodeId();
    }

    public void registerNode() {
        Torrent.RegistrationRequest registrationRequest = Torrent.RegistrationRequest.newBuilder()
                .setOwner(this.getNodeId().getOwner())
                .setIndex(this.getNodeId().getIndex())
                .setPort(this.getNodeId().getPort())
                .build();
        Torrent.Message message = Torrent.Message.newBuilder()
                .setType(Torrent.Message.Type.REGISTRATION_REQUEST)
                .setRegistrationRequest(registrationRequest)
                .build();
        try {
            Socket socket = new Socket(Constants.HOST, Constants.HUB_PORT);
            OutputStream outputStream = socket.getOutputStream();
            Utils.writeMessage(message, outputStream);
            InputStream inputStream = socket.getInputStream();
            Torrent.Message response = Utils.readMessage(inputStream);
            System.out.println(response.getRegistrationResponse().getStatus() + " " + response.getRegistrationResponse().getErrorMessage());
            if (response.getRegistrationResponse().getStatus() == Torrent.Status.SUCCESS) {
                this.registered = true;
            }
            socket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void awaitRequests() {
        ServerSocket serverSocket;
        try {
            serverSocket = new ServerSocket(this.getNodeId().getPort());
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        while (true) {
            try {
                Socket socket = serverSocket.accept();
                InputStream inputStream = socket.getInputStream();
                Torrent.Message message = Utils.readMessage(inputStream);
                Torrent.Message responseMessage = this.handleRequest(message);
                Utils.writeMessage(responseMessage, socket.getOutputStream());
                socket.close();
            } catch (SocketTimeoutException e) {
                System.out.println("Socket time out");
                break;
            } catch (IOException e) {
                e.printStackTrace();
                break;
            }
        }
    }

    private Torrent.Message handleRequest(Torrent.Message message) {
        switch (message.getType()) {
            case UPLOAD_REQUEST:
                return uploadService.handleUploadRequest(message.getUploadRequest());
            case REPLICATE_REQUEST:
                return replicateService.handleReplicateRequest(message.getReplicateRequest());
            case CHUNK_REQUEST:
                return chunkService.handleChunkRequest(message.getChunkRequest());
            case LOCAL_SEARCH_REQUEST:
                return localSearchService.handleLocalSearchRequest(message.getLocalSearchRequest());
            case SEARCH_REQUEST:
                return searchService.handleSearchRequest(message.getSearchRequest());
            default:
                return downloadService.handleDownloadRequest(message.getDownloadRequest());
        }
    }
}
