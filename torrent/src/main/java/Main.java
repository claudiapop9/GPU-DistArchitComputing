import services.NodeService;

public class Main {
    public static void main(String[] args) {
        int port = Integer.parseInt(args[0]);
        int id = port % 5000 - 3;
        NodeService nodeService = new NodeService(port, id);
        int counter = 0;
        while (!nodeService.isRegistered()) {
            System.out.println("Trying to register "+ (++counter)  +" time...");
            nodeService.registerNode();
        }
        nodeService.awaitRequests();
    }
}
