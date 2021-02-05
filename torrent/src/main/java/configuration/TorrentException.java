package configuration;

import entities.Torrent;

public class TorrentException extends Exception {
    private Torrent.Status status;

    public TorrentException(String message, Torrent.Status status) {
        super(message);
        this.status = status;
    }

    public Torrent.Status getStatus() {
        return status;
    }
}
