// use twelite mwx c++ template library
#include <TWELITE>
#include <NWK_SIMPLE>

/*** Config part */
// application ID
const uint32_t APP_ID = 0x1234abcd;

// channel
const uint8_t CHANNEL = 13;

/*** application defs */
const int MSG_LEN = 4;

/*** setup procedure (run once at cold boot) */
void setup() {

	// the twelite main class
	the_twelite
		<< TWENET::appid(APP_ID)    // set application ID (identify network group)
		<< TWENET::channel(CHANNEL) // set channel (pysical channel)
		<< TWENET::rx_when_idle();  // open receive circuit (if not set, it can't listen packts from others)

	// Register Network
	auto&& nwksmpl = the_twelite.network.use<NWK_SIMPLE>();
	nwksmpl << NWK_SIMPLE::logical_id(0x00) // set Logical ID. (0xFE means a child device with no ID)
	        << NWK_SIMPLE::repeat_max(0);   // can repeat a packet up to three times. (being kind of a router)

	the_twelite.begin(); // start twelite!

}

/*** loop procedure (called every event) */
void loop() {

}



void on_rx_packet(packet_rx& rx, bool_t &handled) {
	// rx >> Serial; // debugging (display longer packet information)

	uint8_t msg[MSG_LEN];
	uint32_t timestamp;

	// expand packet payload (shall match with sent packet data structure, see pack_bytes())
	expand_bytes(rx.get_payload().begin(), rx.get_payload().end()
				, msg       // 4bytes of msg
							//   also can be -> std::make_pair(&msg[0], MSG_LEN)
				, timestamp // 4bytes of timestamp
	);
	
	// display the packet
	Serial << format("<RX ad=%x/lq=%d/ln=%d/sq=%d:" // note: up to 4 args!
				, rx.get_psRxDataApp()->u32SrcAddr
				, rx.get_lqi()
				, rx.get_length()
				, rx.get_psRxDataApp()->u8Seq
				)
			<< format(" %s TS=%dms>" // note: up to 4 args!
				, msg
				, timestamp
				)
			<< mwx::crlf
			<< mwx::flush;
}
