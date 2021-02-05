package services;

import com.google.protobuf.InvalidProtocolBufferException;
import configuration.Constants;
import configuration.ErrorMessages;
import configuration.TorrentException;
import entities.Torrent;
import utils.Utils;

import java.io.IOException;
import java.net.Socket;

class SubnetService {

    static Torrent.SubnetResponse getSubnets(int subnetId) throws TorrentException {
        try {
            Socket socket = new Socket(Constants.HOST, Constants.HUB_PORT);
            Torrent.SubnetRequest subnetRequest = Torrent.SubnetRequest.newBuilder()
                    .setSubnetId(subnetId)
                    .build();
            Torrent.Message message = Torrent.Message.newBuilder()
                    .setType(Torrent.Message.Type.SUBNET_REQUEST)
                    .setSubnetRequest(subnetRequest)
                    .build();
            Utils.writeMessage(message, socket.getOutputStream());
            Torrent.Message responseMessage = Utils.readMessage(socket.getInputStream());
            socket.close();
            if (responseMessage.getType() != Torrent.Message.Type.SUBNET_RESPONSE) {
                throw new TorrentException(ErrorMessages.WRONG_TYPE + Torrent.Message.Type.SUBNET_RESPONSE, Torrent.Status.PROCESSING_ERROR);
            }
            return responseMessage.getSubnetResponse();
        } catch (InvalidProtocolBufferException e) {
            throw new TorrentException(ErrorMessages.INVALID_FORMAT, Torrent.Status.PROCESSING_ERROR);
        } catch (IOException e) {
            e.printStackTrace();
            throw new TorrentException(ErrorMessages.NETWORK_ERROR + " " + Torrent.Message.Type.SUBNET_REQUEST, Torrent.Status.PROCESSING_ERROR);
        }
    }
}